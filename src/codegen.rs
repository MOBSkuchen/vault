use inkwell::module::Module;
use inkwell::OptimizationLevel;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple};
use rand::{rng, Rng};
use rand::distr::Alphanumeric;


pub struct Codegen {
    target_triple: TargetTriple,
    optimization_level: OptimizationLevel,
    reloc_mode: RelocMode,
    code_model: CodeModel
}

impl Codegen {
    pub fn new(target_triple: Option<TargetTriple>,
               optimization_level: Option<OptimizationLevel>,
               reloc_mode: Option<RelocMode>,
               code_model: Option<CodeModel>) -> Self {
        Target::initialize_all(&InitializationConfig::default());
        Self { target_triple: target_triple.unwrap_or(TargetMachine::get_default_triple()),
            optimization_level: optimization_level.unwrap_or(OptimizationLevel::Aggressive),
            reloc_mode: reloc_mode.unwrap_or(RelocMode::Default),
            code_model: code_model.unwrap_or(CodeModel::Default)
        }
    }

    fn create_target_machine(&self) -> TargetMachine {
        let target = Target::from_triple(&self.target_triple).unwrap();
        let target_machine = target
            .create_target_machine(
                &self.target_triple,
                "generic",
                "",
                self.optimization_level,
                self.reloc_mode,
                self.code_model
            )
            .expect("Failed to create target machine");
        target_machine
    }

    pub fn optimize(&self, module: &Module) {
        let pb = PassBuilderOptions::create();
        // pb.set_debug_logging(true);
        module.run_passes("default<O3>", &self.create_target_machine(), pb).expect("Failed to optimize module!")
    }

    fn gen_dest_path() -> String {
        rng()
            .sample_iter(&Alphanumeric)
            .take(30)
            .map(char::from)
            .collect()
    }

    pub fn gen_obj(&self, module: &Module, path: Option<String>) -> String {
        let mut path = path.unwrap_or(Self::gen_dest_path());
        path += ".o";
        self.create_target_machine().write_to_file(module, FileType::Object, (&path).as_ref()).expect("TODO: panic message");
        path
    }

    pub fn gen_asm(&self, module: &Module, path: Option<String>) -> String {
        let mut path = path.unwrap_or(Self::gen_dest_path());
        path += ".asm";
        self.create_target_machine().write_to_file(module, FileType::Assembly, (&path).as_ref()).expect("TODO: panic message");
        path
    }
}