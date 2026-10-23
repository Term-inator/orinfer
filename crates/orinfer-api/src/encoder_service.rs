//! Shared bounded HTTP ingress and an owner-thread GPU lifecycle for encoders.
use serde_json::{Value, json};
use std::time::Instant;
type Result<T> = std::result::Result<T, String>;
pub(crate) trait Backend {
    fn respond(&mut self, request: Value) -> Result<Value>;
}
#[derive(Clone)]
struct Service {
    sender: tokio::sync::mpsc::Sender<Job>,
    slots: std::sync::Arc<tokio::sync::Semaphore>,
    model: std::sync::Arc<str>,
    key: Option<std::sync::Arc<str>>,
    timeout_ms: u64,
}
struct Job {
    _slot: tokio::sync::OwnedSemaphorePermit,
    request: Value,
    result: tokio::sync::oneshot::Sender<Result<Value>>,
    deadline: Instant,
}

pub(crate) fn serve<B, F>(
    settings: crate::ServerConfig,
    routes: Vec<&'static str>,
    load: F,
) -> Result<()>
where
    B: Backend + 'static,
    F: FnOnce(&crate::ServerConfig) -> Result<B> + Send + 'static,
{
    if !settings.default_request_params.is_empty() {
        return Err(
            "Encoder services accept request fields directly; Chat defaults are unsupported".into(),
        );
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(serve_async(settings, routes, load))
}
async fn serve_async<B, F>(
    settings: crate::ServerConfig,
    routes: Vec<&'static str>,
    load: F,
) -> Result<()>
where
    B: Backend + 'static,
    F: FnOnce(&crate::ServerConfig) -> Result<B> + Send + 'static,
{
    use axum::{
        Router,
        extract::DefaultBodyLimit,
        routing::{get, post},
    };
    let (sender, mut receiver) = tokio::sync::mpsc::channel::<Job>(4);
    let (ready, initialized) = tokio::sync::oneshot::channel();
    let state = Service {
        sender,
        slots: std::sync::Arc::new(tokio::sync::Semaphore::new(8)),
        model: settings.model.clone().into(),
        key: std::env::var("ORINFER_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .map(Into::into),
        timeout_ms: settings.limits.output_ms,
    };
    let listen = settings.listen.clone();
    let worker = std::thread::Builder::new()
        .name("orinfer-encoder-gpu".into())
        .spawn(move || {
            let initialize = || -> Result<_> {
                let lock = crate::manager::locked_file(&settings.gpu_lock)?;
                let backend = load(&settings)?;
                struct Loaded<B> {
                    backend: B,
                    _lock: std::fs::File,
                }
                Ok(Loaded {
                    backend,
                    _lock: lock,
                })
            };
            match initialize() {
                Err(error) => {
                    let _ = ready.send(Err(error));
                }
                Ok(mut loaded) => {
                    if ready.send(Ok(())).is_err() {
                        return;
                    }
                    while let Some(job) = receiver.blocking_recv() {
                        if job.result.is_closed() {
                            continue;
                        }
                        let result = if Instant::now() >= job.deadline {
                            Err("Encoder request expired in queue".into())
                        } else {
                            loaded.backend.respond(job.request)
                        };
                        let _ = job.result.send(result);
                    }
                }
            }
        })
        .map_err(|e| e.to_string())?;
    let initialized = initialized
        .await
        .map_err(|_| "Encoder GPU worker disconnected during startup".to_owned())
        .and_then(|r| r);
    if let Err(error) = initialized {
        drop(state);
        let _ = worker.join();
        return Err(error);
    }
    let result = async {
        let mut router = Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models));
        for route in routes {
            router = router.route(route, post(inference_request));
        }
        let router = router
            .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
            .with_state(state.clone());
        async fn shutdown() {
            #[cfg(unix)]
            {
                let mut terminate =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("SIGTERM listener");
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
        if let Some(path) = listen.strip_prefix("unix:") {
            let listener = tokio::net::UnixListener::bind(path).map_err(|e| e.to_string())?;
            eprintln!(
                "API READY encoders at {listen}; one complete independent sequence at a time"
            );
            axum::serve(listener, router)
                .with_graceful_shutdown(shutdown())
                .await
                .map_err(|e| e.to_string())
        } else {
            let listener = tokio::net::TcpListener::bind(&listen)
                .await
                .map_err(|e| e.to_string())?;
            eprintln!(
                "API READY encoders at {listen}; one complete independent sequence at a time"
            );
            axum::serve(listener, router)
                .with_graceful_shutdown(shutdown())
                .await
                .map_err(|e| e.to_string())
        }
    }
    .await;
    drop(state);
    tokio::task::spawn_blocking(move || worker.join())
        .await
        .map_err(|e| e.to_string())?
        .map_err(|_| "Encoder GPU worker panicked".to_owned())?;
    result
}

fn authorized(state: &Service, headers: &axum::http::HeaderMap) -> bool {
    state.key.as_ref().is_none_or(|key| {
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            == Some(key.as_ref())
    })
}
async fn health(axum::extract::State(state): axum::extract::State<Service>) -> axum::Json<Value> {
    axum::Json(json!({"status":"ready","model":state.model.as_ref()}))
}
fn failure(status: axum::http::StatusCode, message: impl Into<String>) -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        status,
        axum::Json(json!({"error":{"message":message.into(),"type":"invalid_request_error"}})),
    )
        .into_response()
}
async fn models(
    axum::extract::State(state): axum::extract::State<Service>,
    headers: axum::http::HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if !authorized(&state, &headers) {
        return failure(axum::http::StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    axum::Json(json!({"object":"list","data":[{"id":state.model.as_ref(),"object":"model","owned_by":"orinfer"}]})).into_response()
}
async fn inference_request(
    axum::extract::State(state): axum::extract::State<Service>,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> axum::response::Response {
    use axum::{http::StatusCode, response::IntoResponse};
    if !authorized(&state, &headers) {
        return failure(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    let Ok(slot) = std::sync::Arc::clone(&state.slots).try_acquire_owned() else {
        return failure(StatusCode::TOO_MANY_REQUESTS, "Encoder ingress is full");
    };
    let bytes = match axum::body::to_bytes(body, 32 * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(e) => return failure(StatusCode::PAYLOAD_TOO_LARGE, e.to_string()),
    };
    let request = match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) => value,
        Err(e) => return failure(StatusCode::BAD_REQUEST, e.to_string()),
    };
    drop(bytes);
    let (result, receiver) = tokio::sync::oneshot::channel();
    let timeout = std::time::Duration::from_millis(state.timeout_ms);
    if state
        .sender
        .try_send(Job {
            _slot: slot,
            request,
            result,
            deadline: Instant::now() + timeout,
        })
        .is_err()
    {
        return failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Encoder request queue is full",
        );
    }
    match tokio::time::timeout(timeout, receiver).await {
        Ok(Ok(Ok(value))) => axum::Json(value).into_response(),
        Ok(Ok(Err(error))) => failure(StatusCode::BAD_REQUEST, error),
        Ok(Err(_)) => failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Encoder GPU worker disconnected",
        ),
        Err(_) => failure(StatusCode::GATEWAY_TIMEOUT, "Encoder request timed out"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn encoder_health_matches_the_model_manager_readiness_contract() {
        let (sender, _receiver) = tokio::sync::mpsc::channel(1);
        let state = Service {
            sender,
            slots: std::sync::Arc::new(tokio::sync::Semaphore::new(1)),
            model: "test-rerank".into(),
            key: Some("test-key".into()),
            timeout_ms: 10000,
        };
        let axum::Json(value) = health(axum::extract::State(state)).await;
        assert_eq!(value, json!({"status":"ready","model":"test-rerank"}));
    }
    #[tokio::test]
    async fn cancelled_encoder_handler_retains_its_ingress_permit_until_job_drops() {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        let slots = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let state = Service {
            sender,
            slots: std::sync::Arc::clone(&slots),
            model: "embed".into(),
            key: None,
            timeout_ms: 10000,
        };
        let handler = tokio::spawn(inference_request(
            axum::extract::State(state),
            axum::http::HeaderMap::new(),
            axum::body::Body::from(r#"{"model":"embed","input":"hello"}"#),
        ));
        let job = receiver.recv().await.unwrap();
        assert_eq!(job.request["input"], "hello");
        assert_eq!(slots.available_permits(), 0);
        handler.abort();
        let _ = handler.await;
        assert!(job.result.is_closed());
        assert_eq!(slots.available_permits(), 0);
        drop(job);
        assert_eq!(slots.available_permits(), 1);
    }
}
