use inkwell::module::Module;
use inkwell::OptimizationLevel;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple};
use crate::OptLevel;

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

    pub fn optimize(&self, module: &Module, opt_level: &OptLevel) {
        let pb = PassBuilderOptions::create();
        // pb.set_debug_logging(true);
        let pass = match opt_level {
            OptLevel::Null => "default<O0>",
            OptLevel::One => "default<O1>",
            OptLevel::Two => "default<O2>",
            OptLevel::Full => "default<O3>"
        };
        module.run_passes(pass, &self.create_target_machine(), pb).expect("Failed to optimize module!")
    }

    pub fn gen_obj(&self, module: &Module, path: String) -> String {
        self.create_target_machine().write_to_file(module, FileType::Object, path.as_ref()).expect("Failed to write obj to file");
        path
    }

    pub fn gen_asm(&self, module: &Module, path: String) -> String {
        self.create_target_machine().write_to_file(module, FileType::Assembly, path.as_ref()).expect("Failed to write assembly to file");
        path
    }

    pub fn gen_ir(&self, module: &Module, path: String) -> String {
        module.print_to_file(&path).expect("Failed to write module to file");
        path
    }

    pub fn gen_bc(&self, module: &Module, path: String) -> String {
        module.write_bitcode_to_path(&path);
        path
    }
}