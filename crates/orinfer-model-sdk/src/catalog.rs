//! Compiled implementations are independent of tensor roles and model layers.
use crate::{
    artifact::{Argument, FileIdentity, Kernel, Result},
    operators::dynamic::{DynamicBatchKernel, RowArgument, RowExpression},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterKind {
    Buffer,
    BufferSlice,
    I32,
    U32,
    I64,
    U64,
    F32,
}
impl ParameterKind {
    pub fn of(argument: &Argument) -> Self {
        match argument {
            Argument::Buffer { .. } => Self::Buffer,
            Argument::BufferSlice { .. } => Self::BufferSlice,
            Argument::I32 { .. } => Self::I32,
            Argument::U32 { .. } => Self::U32,
            Argument::I64 { .. } => Self::I64,
            Argument::U64 { .. } => Self::U64,
            Argument::F32 { .. } => Self::F32,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch: Option<crate::artifact::Launch>,
    pub module: usize,
    pub source: usize,
    pub host_abi: usize,
    pub symbol: String,
    #[serde(default)]
    pub block: [u32; 3],
    #[serde(default)]
    pub shared_memory_bytes: u32,
    #[serde(default)]
    pub cooperative: bool,
    pub parameters: Vec<ParameterKind>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub readonly_arguments: Vec<bool>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicDefinition {
    pub capacity: usize,
    pub grid: [RowExpression; 3],
    pub arguments: Vec<RowArgument>,
}
impl DynamicDefinition {
    pub fn bind(&self, name: &str) -> DynamicBatchKernel {
        DynamicBatchKernel {
            name: name.into(),
            capacity: self.capacity,
            grid: self.grid.clone(),
            arguments: self.arguments.clone(),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub assets: Vec<FileIdentity>,
    pub definitions: Vec<Definition>,
    pub dynamic: Vec<DynamicDefinition>,
    /// Seals the generated native role/implementation mapping, not weight contents.
    pub binding_signature: String,
}
impl Catalog {
    pub fn validate(&self) -> Result<()> {
        if self.binding_signature.len() != 64
            || !self
                .binding_signature
                .bytes()
                .all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase())
            || self.definitions.is_empty()
        {
            return Err("Invalid native binding signature or empty kernel catalog".into());
        }
        for d in &self.definitions {
            if [d.module, d.source, d.host_abi]
                .iter()
                .any(|&i| i >= self.assets.len())
                || d.symbol.is_empty()
                || d.parameters.len() > 256
                || (!d.readonly_arguments.is_empty()
                    && d.readonly_arguments.len() != d.parameters.len())
            {
                return Err("Invalid compiled kernel definition".into());
            }
            crate::artifact::validate_geometry(
                d.launch.as_ref(),
                None,
                d.block,
                d.shared_memory_bytes,
                d.cooperative,
            )?;
        }
        Ok(())
    }
    pub fn validate_target(&self, target: &orinfer_device::Target) -> Result<()> {
        self.validate()?;
        target.validate()?;
        for definition in &self.definitions {
            let backend = definition.launch.as_ref().map_or(
                orinfer_device::Backend::Cuda,
                crate::artifact::Launch::backend,
            );
            if backend != target.backend {
                return Err("Kernel catalog backend differs from package target".into());
            }
        }
        Ok(())
    }
    pub fn bind(
        &self,
        definition: usize,
        name: String,
        grid: [u32; 3],
        args: Vec<Argument>,
    ) -> Result<Kernel> {
        let d = self
            .definitions
            .get(definition)
            .ok_or("Unknown compiled kernel definition")?;
        crate::artifact::validate_geometry(
            d.launch.as_ref(),
            Some(grid),
            d.block,
            d.shared_memory_bytes,
            d.cooperative,
        )?;
        if d.parameters.len() != args.len()
            || d.parameters
                .iter()
                .zip(&args)
                .any(|(&kind, a)| kind != ParameterKind::of(a))
        {
            return Err(format!(
                "{name}: native binding differs from compiled parameter ABI"
            ));
        }
        let asset = |i| {
            self.assets
                .get(i)
                .cloned()
                .ok_or_else(|| "Unknown kernel asset".to_owned())
        };
        Ok(Kernel {
            launch: d.launch.clone(),
            name,
            grid,
            args,
            module: asset(d.module)?,
            source: asset(d.source)?,
            host_abi: asset(d.host_abi)?,
            symbol: d.symbol.clone(),
            block: d.block,
            shared_memory_bytes: d.shared_memory_bytes,
            cooperative: d.cooperative,
            readonly_arguments: d.readonly_arguments.clone(),
        })
    }
}

/// Build-time generated role references. Offsets remain explicit and checked by
/// the common runtime against the model's physical tensor layout.
pub enum RoleArgument {
    Buffer(usize),
    BufferSlice(usize, usize),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(u32),
}
impl RoleArgument {
    pub fn bind(&self, roles: &[&str]) -> Result<Argument> {
        let role = |i: usize| {
            roles
                .get(i)
                .map(|s| (*s).to_owned())
                .ok_or_else(|| "Unknown native tensor role".to_owned())
        };
        Ok(match *self {
            Self::Buffer(i) => Argument::Buffer { name: role(i)? },
            Self::BufferSlice(i, offset) => Argument::BufferSlice {
                name: role(i)?,
                offset,
            },
            Self::I32(value) => Argument::I32 { value },
            Self::U32(value) => Argument::U32 { value },
            Self::I64(value) => Argument::I64 { value },
            Self::U64(value) => Argument::U64 { value },
            Self::F32(bits) => Argument::F32 {
                value: f32::from_bits(bits),
            },
        })
    }
}
/// One local specialization, shared by every layer with this implementation ABI.
pub struct BindingVariant {
    pub suffix: &'static str,
    pub definition: usize,
    pub grid: [u32; 3],
    pub arguments: &'static [usize],
    pub dynamic: Option<usize>,
    pub plan_contract: bool,
}

pub struct LayerBinding {
    pub layer: Option<usize>,
    pub parameters: usize,
}

/// Same invocation slot across layers. Only the parameter table varies.
pub struct BindingTemplate {
    pub prefix: &'static str,
    pub suffix: &'static str,
    pub choice: usize,
    pub layers: &'static [LayerBinding],
}

pub struct BindingRegistry {
    pub roles: &'static [&'static str],
    pub parameters: &'static [&'static [RoleArgument]],
    pub choices: &'static [&'static [BindingVariant]],
    pub templates: &'static [BindingTemplate],
}

pub struct BoundBinding<'a> {
    registry: &'a BindingRegistry,
    template: &'a BindingTemplate,
    layer: &'a LayerBinding,
    pub variant: &'a BindingVariant,
}

impl BoundBinding<'_> {
    pub fn name(&self) -> String {
        let layer = self.layer.layer.map(|i| i.to_string()).unwrap_or_default();
        format!(
            "{}{layer}{}{}",
            self.template.prefix, self.template.suffix, self.variant.suffix
        )
    }

    pub fn arguments(&self) -> Result<Vec<Argument>> {
        let parameters = self
            .registry
            .parameters
            .get(self.layer.parameters)
            .ok_or("Unknown native layer parameters")?;
        self.variant
            .arguments
            .iter()
            .map(|&i| {
                parameters
                    .get(i)
                    .ok_or("Unknown native parameter slot")?
                    .bind(self.registry.roles)
            })
            .collect()
    }
}

impl BindingRegistry {
    pub fn iter(&self) -> Result<impl Iterator<Item = BoundBinding<'_>>> {
        if self
            .templates
            .iter()
            .any(|t| t.choice >= self.choices.len())
        {
            return Err("Unknown native specialization table".into());
        }
        Ok(self.templates.iter().flat_map(move |template| {
            template.layers.iter().flat_map(move |layer| {
                self.choices[template.choice]
                    .iter()
                    .map(move |variant| BoundBinding {
                        registry: self,
                        template,
                        layer,
                        variant,
                    })
            })
        }))
    }

    /// Resolve a single continuation operand without expanding the registry.
    pub fn find(&self, name: &str) -> Result<BoundBinding<'_>> {
        for template in self.templates {
            let Some(mut tail) = name.strip_prefix(template.prefix) else {
                continue;
            };
            let layer = if template.layers.first().is_some_and(|l| l.layer.is_some()) {
                let digits = tail.bytes().take_while(u8::is_ascii_digit).count();
                if digits == 0 {
                    continue;
                }
                let (number, rest) = tail.split_at(digits);
                if number.len() > 1 && number.starts_with('0') {
                    continue;
                }
                tail = rest;
                Some(
                    number
                        .parse::<usize>()
                        .map_err(|_| "Invalid native layer index")?,
                )
            } else {
                None
            };
            let Some(tail) = tail.strip_prefix(template.suffix) else {
                continue;
            };
            let Some(layer) = template.layers.iter().find(|l| l.layer == layer) else {
                continue;
            };
            let choices = self
                .choices
                .get(template.choice)
                .ok_or("Unknown native specialization table")?;
            if let Some(variant) = choices.iter().find(|v| v.suffix == tail) {
                return Ok(BoundBinding {
                    registry: self,
                    template,
                    layer,
                    variant,
                });
            }
        }
        Err(format!("Missing native continuation binding {name}"))
    }
}

#[cfg(test)]
mod binding_tests {
    use super::*;

    #[test]
    fn native_ascend_binding_retains_launch_and_checks_package_target() {
        let asset = FileIdentity {
            file: "kernel.bin".into(),
            sha256: "0".repeat(64),
            tensor: None,
        };
        let catalog = Catalog {
            assets: vec![asset],
            binding_signature: "0".repeat(64),
            dynamic: vec![],
            definitions: vec![Definition {
                launch: Some(crate::artifact::Launch::Ascend {
                    block_dim: 24,
                    workspace_bytes: 8192,
                }),
                module: 0,
                source: 0,
                host_abi: 0,
                symbol: "projection".into(),
                block: [0; 3],
                shared_memory_bytes: 0,
                cooperative: false,
                parameters: vec![ParameterKind::U32],
                readonly_arguments: vec![],
            }],
        };
        let target = orinfer_device::Target::from_package("ascend910b").unwrap();
        catalog.validate_target(&target).unwrap();
        assert!(
            catalog
                .validate_target(&orinfer_device::Target::from_package("sm_87").unwrap())
                .is_err()
        );
        let kernel = catalog
            .bind(
                0,
                "projection".into(),
                [0; 3],
                vec![Argument::U32 { value: 16 }],
            )
            .unwrap();
        kernel.validate_launch(&target).unwrap();
        assert!(matches!(
            kernel.launch,
            Some(crate::artifact::Launch::Ascend {
                block_dim: 24,
                workspace_bytes: 8192
            })
        ));
        assert!(
            catalog
                .bind(
                    0,
                    "projection".into(),
                    [1; 3],
                    vec![Argument::U32 { value: 16 }]
                )
                .is_err()
        );
        assert!(
            catalog
                .bind(
                    0,
                    "projection".into(),
                    [0; 3],
                    vec![Argument::I32 { value: 16 }]
                )
                .is_err()
        );
    }

    static REGISTRY: BindingRegistry = BindingRegistry {
        roles: &["Weight0", "Weight1", "State0", "State1"],
        parameters: &[
            &[
                RoleArgument::Buffer(0),
                RoleArgument::BufferSlice(2, 64),
                RoleArgument::I32(-7),
                RoleArgument::F32(0x80000000),
            ],
            &[
                RoleArgument::Buffer(1),
                RoleArgument::BufferSlice(3, 128),
                RoleArgument::I32(-7),
                RoleArgument::F32(0x80000000),
            ],
        ],
        choices: &[&[
            BindingVariant {
                suffix: "",
                definition: 0,
                grid: [1024, 1, 1],
                arguments: &[0, 1, 2, 3],
                dynamic: Some(0),
                plan_contract: true,
            },
            BindingVariant {
                suffix: "/rows/1-1",
                definition: 1,
                grid: [1, 1, 1],
                arguments: &[1, 0, 3],
                dynamic: None,
                plan_contract: false,
            },
        ]],
        templates: &[BindingTemplate {
            prefix: "target/layer",
            suffix: "/projection",
            choice: 0,
            layers: &[
                LayerBinding {
                    layer: Some(0),
                    parameters: 0,
                },
                LayerBinding {
                    layer: Some(1),
                    parameters: 1,
                },
            ],
        }],
    };

    #[test]
    fn layers_and_local_choices_resolve_identical_order_offsets_and_float_bits() {
        let expanded: Vec<_> = REGISTRY.iter().unwrap().collect();
        assert_eq!(expanded.len(), 4);
        for binding in expanded {
            let name = binding.name();
            let found = REGISTRY.find(&name).unwrap();
            assert_eq!(found.name(), name);
            assert_eq!(found.variant.definition, binding.variant.definition);
            assert_eq!(
                serde_json::to_value(found.arguments().unwrap()).unwrap(),
                serde_json::to_value(binding.arguments().unwrap()).unwrap()
            );
        }
        let binding = REGISTRY.find("target/layer1/projection/rows/1-1").unwrap();
        let args = binding.arguments().unwrap();
        assert!(
            matches!(&args[0], Argument::BufferSlice { name, offset: 128 } if name == "State1")
        );
        assert!(matches!(&args[1], Argument::Buffer { name } if name == "Weight1"));
        assert!(matches!(&args[2], Argument::F32 { value } if value.to_bits() == 0x80000000));
        for bad in [
            "target/layer2/projection",
            "target/layer01/projection",
            "target/layer1/projection/rows/9-9",
        ] {
            assert!(REGISTRY.find(bad).is_err());
        }
    }
}
