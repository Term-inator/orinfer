//! Images-only OpenAI content parts. CPU work runs outside the async executor.
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{ImageReader, imageops::FilterType};
use orinfer_engine::vision::{ImageInput, VisionSpec};
use serde_json::{Value, json};
use std::{
    io::{Cursor, Read},
    time::Duration,
};
type Result<T> = std::result::Result<T, String>;
const MAX_BYTES: usize = 24 * 1024 * 1024;
const MIN_PIXELS: usize = 256 * 256;
const MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

fn load(url: &str, preparation: &super::preparation::Context) -> Result<Vec<u8>> {
    preparation.checkpoint(MAX_BYTES)?;
    if let Some(data) = url.strip_prefix("data:") {
        let (mime, payload) = data.split_once(',').ok_or("Invalid image data URI")?;
        if !["image/png;base64", "image/jpeg;base64", "image/webp;base64"].contains(&mime)
            || payload.len() > MAX_BYTES.div_ceil(3) * 4
        {
            return Err("Unsupported or oversized image data URI".into());
        }
        return STANDARD
            .decode(payload)
            .map_err(|e| format!("Image base64: {e}"));
    }
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("Image URL: {e}"))?;
    if !["http", "https"].contains(&parsed.scheme()) {
        return Err("Images require HTTP(S) or a base64 data URI".into());
    }
    let response = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?
        .get(parsed)
        .send()
        .map_err(|e| format!("Image fetch: {e}"))?
        .error_for_status()
        .map_err(|e| format!("Image fetch: {e}"))?;
    if response
        .content_length()
        .is_some_and(|n| n > MAX_BYTES as u64)
    {
        return Err("Image exceeds byte limit".into());
    }
    let mut data = Vec::new();
    response
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    if data.len() > MAX_BYTES {
        return Err("Image exceeds byte limit".into());
    }
    Ok(data)
}
fn round_even(x: f64) -> usize {
    x.round_ties_even() as usize
}
pub fn resize_shape(
    height: usize,
    width: usize,
    factor: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0
        || width == 0
        || height.max(width) as f64 / height.min(width) as f64 > 200.0
        || max_pixels < MIN_PIXELS
    {
        return Err("Invalid image size or aspect ratio".into());
    }
    let (mut h, mut w) = (
        round_even(height as f64 / factor as f64) * factor,
        round_even(width as f64 / factor as f64) * factor,
    );
    let area = (height as f64) * (width as f64);
    if h * w > max_pixels {
        let beta = (area / max_pixels as f64).sqrt();
        h = ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        w = ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if h * w < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / area).sqrt();
        h = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        w = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((h, w))
}
pub fn preprocess(
    data: &[u8],
    v: &VisionSpec,
    detail: &str,
    preparation: &mut super::preparation::Context,
) -> Result<ImageInput> {
    preprocess_with_limit(data, v, detail, None, preparation)
}
fn preprocess_with_limit(
    data: &[u8],
    v: &VisionSpec,
    detail: &str,
    feature_limit: Option<usize>,
    preparation: &mut super::preparation::Context,
) -> Result<ImageInput> {
    if !["auto", "high", "low"].contains(&detail) {
        return Err("image_url.detail must be auto, high or low".into());
    }
    preparation.checkpoint(0)?;
    let (width, height) = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .into_dimensions()
        .map_err(|e| e.to_string())?;
    let source_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or("Image dimensions overflow")?;
    if source_pixels > 16_777_216 {
        return Err("Image exceeds decoded pixel limit".into());
    }
    let factor = v.patch_size * v.merge_size;
    let configured_pixels = if detail == "low" {
        MIN_PIXELS
    } else {
        (v.max_patches * v.patch_size * v.patch_size).min(16_777_216)
    };
    let max_pixels = configured_pixels
        .min(feature_limit.map_or(usize::MAX, |n| n.saturating_mul(factor * factor)));
    if max_pixels < MIN_PIXELS {
        return Err(
            "Too many images for the model's minimum image detail; reduce the image count".into(),
        );
    }
    let (rh, rw) = resize_shape(
        height as usize,
        width as usize,
        v.patch_size * v.merge_size,
        max_pixels,
    )?;
    if feature_limit.is_some_and(|limit| rh / factor * (rw / factor) > limit) {
        return Err("Image aspect ratio exceeds its automatic image budget at minimum detail; reduce the image count".into());
    }
    // Decode may hold RGBA/16-bit source, RGB conversion and resized patch FP32 together.
    preparation.checkpoint(data.len() + source_pixels * 12 + rh * rw * 16)?;
    let mut reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(256 * 1024 * 1024);
    limits.max_image_width = Some(32768);
    limits.max_image_height = Some(32768);
    reader.limits(limits);
    preparation.checkpoint(0)?;
    let decoded = reader
        .decode()
        .map_err(|e| format!("Image decode: {e}"))?
        .to_rgb8();
    let (height, width) = resize_shape(
        decoded.height() as usize,
        decoded.width() as usize,
        v.patch_size * v.merge_size,
        max_pixels,
    )?;
    // Pillow/torchvision's uint8 bicubic path rounds and clips the horizontal
    // pass before the vertical pass. A single floating-point two-axis resize
    // produces different values at sharp high-contrast edges.
    let horizontal = image::imageops::resize(
        &decoded,
        width as u32,
        decoded.height(),
        FilterType::CatmullRom,
    );
    let rgb = image::imageops::resize(
        &horizontal,
        width as u32,
        height as u32,
        FilterType::CatmullRom,
    );
    drop(horizontal);
    drop(decoded);
    let (gh, gw) = (height / v.patch_size, width / v.patch_size);
    let mut pixels = Vec::with_capacity(gh * gw * v.patch_values()?);
    // Merge blocks are contiguous. Channels precede duplicated temporal planes,
    // then the two spatial dimensions of each 16x16 patch.
    for br in 0..gh / v.merge_size {
        for bc in 0..gw / v.merge_size {
            for ir in 0..v.merge_size {
                for ic in 0..v.merge_size {
                    for ch in 0..3 {
                        for _ in 0..v.temporal_patch_size {
                            for y in 0..v.patch_size {
                                for x in 0..v.patch_size {
                                    let px = rgb.get_pixel(
                                        ((bc * v.merge_size + ic) * v.patch_size + x) as u32,
                                        ((br * v.merge_size + ir) * v.patch_size + y) as u32,
                                    )[ch];
                                    pixels.push(px as f32 / 127.5 - 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    drop(rgb);
    preparation.retain(pixels.len() * std::mem::size_of::<f32>())?;
    let input = ImageInput {
        grid_height: gh,
        grid_width: gw,
        pixels,
    };
    v.feature_count(&input)?;
    Ok(input)
}
pub fn messages(
    mut messages: Vec<Value>,
    spec: Option<&VisionSpec>,
    preparation: &mut super::preparation::Context,
) -> Result<(Vec<Value>, Vec<ImageInput>)> {
    let mut descriptors = Vec::new();
    for message in &messages {
        if let Some(parts) = message["content"].as_array() {
            for part in parts.iter().filter(|p| p["type"] == "image_url") {
                if message["role"] != "user" {
                    return Err("Image parts are supported in user messages".into());
                }
                let url = part["image_url"]["url"]
                    .as_str()
                    .ok_or("image_url requires url")?;
                let detail = part["image_url"]
                    .get("detail")
                    .map(|v| v.as_str().ok_or("Invalid image detail"))
                    .transpose()?
                    .unwrap_or("auto");
                if !["auto", "high", "low"].contains(&detail) {
                    return Err("image_url.detail must be auto, high or low".into());
                }
                descriptors.push((url, detail));
            }
        }
    }
    let count = descriptors.len();
    let mut prepared: Vec<Option<ImageInput>> =
        std::iter::repeat_with(|| None).take(count).collect();
    if count > 0 {
        let v = spec.ok_or("This model manifest has no vision encoder")?;
        let factor = v.patch_size * v.merge_size;
        let minimum = MIN_PIXELS.div_ceil(factor * factor);
        if count > v.max_features / minimum {
            return Err(format!(
                "Images exceed feature capacity at minimum image detail: {count} images, {} available features; reduce the image count",
                v.max_features
            ));
        }
        let mut features = 0usize;
        // Honor explicit detail first, then divide the remaining capacity among
        // automatic images. Output slots preserve original message/image order.
        for (i, &(url, detail)) in descriptors
            .iter()
            .enumerate()
            .filter(|(_, (_, d))| *d != "auto")
        {
            let image = preprocess(&load(url, preparation)?, v, detail, preparation)?;
            features = features
                .checked_add(v.feature_count(&image)?)
                .ok_or("Image feature count overflow")?;
            if features > v.max_features {
                return Err(format!(
                    "Images exceed feature capacity: explicit detail requires {features} features, model limit {}; use auto/low detail or fewer images",
                    v.max_features
                ));
            }
            prepared[i] = Some(image);
        }
        let automatic = descriptors
            .iter()
            .filter(|(_, detail)| *detail == "auto")
            .count();
        if let Some(share) = (v.max_features - features).checked_div(automatic) {
            if share < minimum {
                return Err(format!(
                    "Images exceed feature capacity: explicit detail leaves {} features for {automatic} automatic images; use auto/low detail or fewer images",
                    v.max_features - features
                ));
            }
            for (i, &(url, detail)) in descriptors
                .iter()
                .enumerate()
                .filter(|(_, (_, d))| *d == "auto")
            {
                let image = preprocess_with_limit(
                    &load(url, preparation)?,
                    v,
                    detail,
                    Some(share),
                    preparation,
                )?;
                features = features
                    .checked_add(v.feature_count(&image)?)
                    .ok_or("Image feature count overflow")?;
                prepared[i] = Some(image);
            }
            if features > v.max_features {
                return Err("Images exceed feature capacity".into());
            }
        }
    }
    drop(descriptors);
    let mut images = Vec::with_capacity(count);
    for message in &mut messages {
        let Some(parts) = message["content"].as_array() else {
            continue;
        };
        let mut content = String::new();
        for part in parts {
            preparation.checkpoint(0)?;
            match part["type"].as_str() {
                Some("text") => {
                    content.push_str(part["text"].as_str().ok_or("Text part needs text")?)
                }
                Some("audio") => content.push_str("<|audio_start|><|audio_pad|><|audio_end|>"),
                Some("image_url") => {
                    images.push(
                        prepared[images.len()]
                            .take()
                            .ok_or("Missing prepared image")?,
                    );
                    content.push_str(MARKER);
                }
                _ => return Err("Supported content parts: text and image_url".into()),
            }
        }
        message["content"] = json!(content);
    }
    Ok((messages, images))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Needs Transformers-generated image references"]
    fn checkpoint_image_processor_reference() {
        let root = std::path::PathBuf::from(std::env::var("ORINFER_IMAGE_REFERENCE").unwrap());
        let reference: Value =
            serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
        let spec: VisionSpec = serde_json::from_value(reference["vision"].clone()).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let input = preprocess(
                &std::fs::read(root.join(case["image"].as_str().unwrap())).unwrap(),
                &spec,
                "auto",
                &mut crate::server::preparation::Context::unbounded(),
            )
            .unwrap();
            assert_eq!(
                input.grid_height,
                case["grid"][1].as_u64().unwrap() as usize
            );
            assert_eq!(input.grid_width, case["grid"][2].as_u64().unwrap() as usize);
            let bytes = std::fs::read(root.join(case["pixels"].as_str().unwrap())).unwrap();
            let expected: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(input.pixels.len(), expected.len());
            let (mut square, mut baseline, mut maximum) = (0.0f64, 0.0f64, 0.0f32);
            for (&a, &b) in input.pixels.iter().zip(&expected) {
                square += (a - b) as f64 * (a - b) as f64;
                baseline += b as f64 * b as f64;
                maximum = maximum.max((a - b).abs());
            }
            let relative = (square / baseline).sqrt();
            eprintln!(
                "{} resize relative L2 {relative:.6}, max {maximum:.6}",
                case["image"]
            );
            assert!(relative < 0.02 && maximum < 0.05);
        }
    }
    #[test]
    #[ignore = "Allocates a real 576 MiB multi-image request; run with sufficient host RAM"]
    fn many_images_share_the_configured_preprocessing_pool() {
        let spec: VisionSpec = serde_json::from_value(json!({"hidden":5120,"patch_size":16,
            "temporal_patch_size":2,"merge_size":2,"max_patches":32768,"max_features":32768,
            "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
            "mrope_positions":"r","plans":[{"patches":32768,"program":"v"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12}))
        .unwrap();
        let image = image::RgbImage::from_pixel(2048, 1024, image::Rgb([32, 128, 224]));
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let url = format!(
            "data:image/png;base64,{}",
            STANDARD.encode(encoded.get_ref())
        );
        let request: super::super::ChatRequest = serde_json::from_value(json!({"model":"test","messages":[
            {"role":"user","content":(0..12).map(|_|json!({"type":"image_url","image_url":{"url":url}})).collect::<Vec<_>>()}]})).unwrap();
        let estimate =
            crate::server::preparation::request_memory(&request, 262144, Some(&spec), 2048);
        let mut context = estimate.context(std::sync::Arc::new(
            std::sync::atomic::AtomicBool::new(false),
        ));
        let (_, images) = messages(request.messages, Some(&spec), &mut context).unwrap();
        assert_eq!(images.len(), 12);
        assert_eq!(
            images.iter().map(|i| i.pixels.len() * 4).sum::<usize>(),
            576 << 20
        );
        assert_eq!(
            images
                .iter()
                .map(|i| spec.feature_count(i).unwrap())
                .sum::<usize>(),
            24576
        );
        assert!(
            images
                .iter()
                .all(|i| i.pixels.iter().all(|p| p.is_finite()))
        );
        context.checkpoint(0).unwrap();
    }
    #[test]
    #[ignore = "Real five-image request near the 32768-feature capacity; needs host RAM"]
    fn automatic_large_images_fit_real_model_capacity() {
        let spec: VisionSpec = serde_json::from_value(json!({"hidden":5120,"patch_size":16,
            "temporal_patch_size":2,"merge_size":2,"max_patches":32768,"max_features":32768,
            "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
            "mrope_positions":"r","plans":[{"patches":32768,"program":"v"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12}))
        .unwrap();
        let image = image::RgbImage::from_pixel(4096, 2048, image::Rgb([32, 128, 224]));
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let url = format!(
            "data:image/png;base64,{}",
            STANDARD.encode(encoded.get_ref())
        );
        let request: super::super::ChatRequest = serde_json::from_value(json!({"model":"test","messages":[
            {"role":"user","content":(0..5).map(|_|json!({"type":"image_url","image_url":{"url":url}})).collect::<Vec<_>>()}]})).unwrap();
        assert!(5 * (4096 / 32) * (2048 / 32) > spec.max_features);
        let estimate =
            crate::server::preparation::request_memory(&request, 262144, Some(&spec), 2048);
        let mut context = estimate.context(std::sync::Arc::new(
            std::sync::atomic::AtomicBool::new(false),
        ));
        let (_, images) = messages(request.messages, Some(&spec), &mut context).unwrap();
        assert_eq!(images.len(), 5);
        let features: usize = images.iter().map(|i| spec.feature_count(i).unwrap()).sum();
        assert!(features <= spec.max_features);
        assert!(features > 30000);
        context.checkpoint(0).unwrap();
    }
    #[test]
    fn automatic_images_share_features_and_preserve_input_order() {
        let spec: VisionSpec = serde_json::from_value(json!({"hidden":5120,"patch_size":16,
            "temporal_patch_size":2,"merge_size":2,"max_patches":1024,"max_features":256,
            "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
            "mrope_positions":"r","plans":[{"patches":1024,"program":"v"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12}))
        .unwrap();
        let encode = |size, color| {
            let image = image::RgbImage::from_pixel(size, size, image::Rgb(color));
            let mut output = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(image)
                .write_to(&mut output, image::ImageFormat::Png)
                .unwrap();
            format!(
                "data:image/png;base64,{}",
                STANDARD.encode(output.get_ref())
            )
        };
        let red = encode(512, [255, 0, 0]);
        let blue = encode(512, [0, 0, 255]);
        let part = |url: &str, detail: &str| json!({"type":"image_url","image_url":{"url":url,"detail":detail}});
        let run = |parts: Vec<Value>| {
            messages(
                vec![json!({"role":"user","content":parts})],
                Some(&spec),
                &mut crate::server::preparation::Context::unbounded(),
            )
        };
        let (rendered, images) = run(vec![part(&red, "auto"), part(&blue, "auto")]).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(rendered[0]["content"], format!("{MARKER}{MARKER}"));
        assert!(
            images
                .iter()
                .map(|i| spec.feature_count(i).unwrap())
                .sum::<usize>()
                <= 256
        );
        assert_eq!(images[0].pixels[0], 1.0);
        assert_eq!(images[1].pixels[0], -1.0);
        let (_, reverse) = run(vec![part(&blue, "auto"), part(&red, "auto")]).unwrap();
        assert_eq!(images[0].pixels, reverse[1].pixels);
        assert_eq!(images[1].pixels, reverse[0].pixels);
        let (_, single) = run(vec![part(&red, "auto")]).unwrap();
        assert_eq!(spec.feature_count(&single[0]).unwrap(), 256);
        let error = run(vec![part(&red, "high"), part(&blue, "high")]).unwrap_err();
        assert!(error.contains("feature capacity") && error.contains("512"));
        let small_blue = encode(256, [0, 0, 255]);
        let (_, mixed) = run(vec![part(&red, "auto"), part(&small_blue, "high")]).unwrap();
        assert_eq!((mixed[1].grid_height, mixed[1].grid_width), (16, 16));
        assert_eq!(mixed[0].pixels[0], 1.0);
        assert_eq!(mixed[1].pixels[0], -1.0);
        assert!(
            mixed
                .iter()
                .map(|i| spec.feature_count(i).unwrap())
                .sum::<usize>()
                <= 256
        );
        let error = run((0..5)
            .map(|_| part("https://invalid.invalid/never-fetch", "auto"))
            .collect())
        .unwrap_err();
        assert!(error.contains("minimum image detail"));
    }
    #[test]
    fn patch_order_temporal_duplication_and_image_parts() {
        let spec: VisionSpec = serde_json::from_value(json!({
            "hidden":5120,"patch_size":16,"temporal_patch_size":2,"merge_size":2,
            "max_patches":1024,"max_features":256,"pixels":"p","grid":"g","length":"l",
            "output":"o","features":"f","feature_index":"i","mrope_positions":"r",
            "plans":[{"patches":1024,"program":"v"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12
        }))
        .unwrap();
        let rgb = image::RgbImage::from_fn(256, 256, |x, y| {
            image::Rgb([
                if x / 16 % 2 == 0 { 255 } else { 0 },
                if y / 16 % 2 == 0 { 255 } else { 0 },
                0,
            ])
        });
        let mut data = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(rgb)
            .write_to(&mut data, image::ImageFormat::Png)
            .unwrap();
        let pixels = preprocess(
            data.get_ref(),
            &spec,
            "auto",
            &mut crate::server::preparation::Context::unbounded(),
        )
        .unwrap();
        assert_eq!((pixels.grid_height, pixels.grid_width), (16, 16));
        for (row, (red, green)) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
            .into_iter()
            .enumerate()
        {
            let p = &pixels.pixels[row * 1536..(row + 1) * 1536];
            assert!(p[..512].iter().all(|&v| v == red));
            assert!(p[512..1024].iter().all(|&v| v == green));
            assert!(p[1024..].iter().all(|&v| v == -1.0));
        }
        let url = format!("data:image/png;base64,{}", STANDARD.encode(data.get_ref()));
        let input = vec![
            json!({"role":"user","content":[{"type":"text","text":"A"},{"type":"image_url","image_url":{"url":url}},{"type":"text","text":"B"},{"type":"image_url","image_url":{"url":url}}]}),
        ];
        let (out, images) = messages(
            input.clone(),
            Some(&spec),
            &mut crate::server::preparation::Context::unbounded(),
        )
        .unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(out[0]["content"], format!("A{MARKER}B{MARKER}"));
        assert!(
            messages(
                input,
                None,
                &mut crate::server::preparation::Context::unbounded()
            )
            .is_err()
        );
        assert!(
            preprocess(
                data.get_ref(),
                &spec,
                "invalid",
                &mut crate::server::preparation::Context::unbounded()
            )
            .is_err()
        );
    }
    #[test]
    fn smart_resize_rounding_and_aspect() {
        assert_eq!(resize_shape(256, 256, 32, 1_048_576).unwrap(), (256, 256));
        assert_eq!(resize_shape(80, 112, 32, 1_048_576).unwrap(), (224, 320));
        assert_eq!(
            resize_shape(2000, 1000, 32, 1_048_576).unwrap(),
            (1440, 704)
        );
        assert_eq!(resize_shape(500, 100, 32, 65_536).unwrap(), (576, 128));
        assert!(resize_shape(1, 300, 32, 1_048_576).is_err());
        assert!(
            load(
                "file:///tmp/image.png",
                &crate::server::preparation::Context::unbounded()
            )
            .is_err()
        );
        assert!(
            load(
                "data:video/mp4;base64,AA==",
                &crate::server::preparation::Context::unbounded()
            )
            .is_err()
        );
    }
}
