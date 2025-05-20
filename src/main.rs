use std::fmt::{write, Display, Formatter};
use clap::{Arg, ValueHint};
use inkwell::context::Context;
use inkwell::targets::{TargetMachine, TargetTriple};
use crate::codegen::Codegen;
use crate::comp_errors::CodeResult;
use crate::compiler::Compiler;
use crate::filemanager::FileManager;
use crate::lexer::tokenize;
use crate::parser::Parser;

pub const NAME: &str = env!("CARGO_PKG_NAME");
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

mod parser;
mod compiler;
mod codegen;
mod lexer;
mod comp_errors;
mod filemanager;
mod codeviz;

enum OptLevel {
    Null = 0,
    One = 1,
    Two = 3,
    Full = 4
}

impl Display for OptLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            OptLevel::Full => "full/3",
            OptLevel::Two => "two/2",
            OptLevel::One => "one/1",
            OptLevel::Null => "null/0",
        })
    }
}

enum OutputType {
    Object,
    Asm,
    IR
}

impl Display for OutputType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            OutputType::Object => "object",
            OutputType::Asm => "asm",
            OutputType::IR => "ir"
        })
    }
}

macro_rules! sref {
    ($s:expr) => {
        {let s = $s.to_string();
        &s}
    };
}

impl From<Option<&String>> for OutputType {
    fn from(value: Option<&String>) -> Self {
        match value {
            None => {OutputType::Object}
            Some(s) => {
                let s = s.to_lowercase();
                if s == "object" || s == "o" || s == "obj" {OutputType::Object}
                else if s == "asm" || s == "assembly" {OutputType::Asm}
                else if s == "ir" || s == "ll" {OutputType::IR}
                else {println!("unrecognized output-type `{s}`, defaulting to object"); OutputType::Object}
            }
        }
    }
}

impl From<Option<&String>> for OptLevel {
    fn from(value: Option<&String>) -> Self {
        match value {
            None => {OptLevel::Full}
            Some(s) => {
                if s == "0" {OptLevel::Null}
                else if s == "1" {OptLevel::One}
                else if s == "2" {OptLevel::Two}
                else if s == "3" {OptLevel::Full}
                else {println!("unrecognized optimization level `{s}`, defaulting to Full"); OptLevel::One}

            }
        }
    }
}

struct CompileJobData {
    output: String,
    target_triple: TargetTriple,
    optimization: OptLevel,
    module_id: String,
    output_type: OutputType
}

impl Display for CompileJobData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Module `{}` for `{}` to output file: `{}` (type: {}) with optimization {}", 
               self.module_id, self.target_triple, 
               self.output, self.output_type, 
               self.optimization)
    }
}

fn compile_job(file_manager: &FileManager, compile_job_data: CompileJobData) -> CodeResult<()> {
    let tokens = tokenize(file_manager.get_content())?;
    println!("Compiling `{}` with profile:\n{compile_job_data}", file_manager.input_file);

    let parser = Parser::new(tokens, file_manager);
    let ast = parser.parse(&mut 0)?;

    let context = Context::create();
    let builder = context.create_builder();
    let module = context.create_module(&compile_job_data.module_id);
    let compiler = Compiler::new(&context, &builder, compile_job_data.module_id);

    let module = compiler.comp_ast(module, ast)?;
    let codegen = Codegen::new(Some(compile_job_data.target_triple), None, None, None);
    codegen.optimize(&module, &compile_job_data.optimization);
    let file = match compile_job_data.output_type {
        OutputType::Object => codegen.gen_obj(&module, compile_job_data.output),
        OutputType::Asm => codegen.gen_asm(&module, compile_job_data.output),
        OutputType::IR => codegen.gen_ir(&module, compile_job_data.output),
    };
    println!("Finished writing to `{file}`!");
    Ok(())
}

fn compile(filepath: String, compile_job_data: CompileJobData) -> bool {
    let file_manager_r = FileManager::new_from(filepath);
    if let Err(item) = file_manager_r {
        item.output();
        return true;
    }

    let file_manager = file_manager_r.unwrap();

    let x = compile_job(&file_manager, compile_job_data);
    if let Err(item) = x {
        item.visualize_error(&file_manager);
    }
    false
}

fn main() {
    let tt = TargetMachine::get_default_triple();
    let tt = tt.as_str().to_str().unwrap();
    let matches = clap::Command::new(NAME)
        .about(DESCRIPTION)
        .version(VERSION)
        .author("MOBSkuchen")
        .help_template("
{before-help}{name} {version} \"{about}\"
by {author}
{usage-heading} {usage}

{all-args}{after-help}

Available output types: ASM, IR, OBJECT (default)

This is a temporary build - critical breaking changes WILL occur. Be warned.
")
        .color(clap_builder::ColorChoice::Never)
        .disable_version_flag(true)
        .arg(Arg::new("compile")
            .index(1)
            .help("Compile a file")
            .value_hint(ValueHint::AnyPath)
            .value_name("FILE")
            .action(clap::ArgAction::Set))
        .arg(Arg::new("out-type")
            .long("output-type")
            .alias("ot")
            .help("Set the output type instead of inferring it")
            .value_name("TYPE")
            .action(clap::ArgAction::Set)
            .requires("compile"))
        .arg(Arg::new("optimization")
            .long("optimization")
            .short('O')
            .help("Set optimization level 0-3")
            .value_name("LEVEL")
            .action(clap::ArgAction::Set)
            .requires("compile"))
        .arg(Arg::new("module-id")
            .long("module-id")
            .help("LLVM module id")
            .value_name("ID")
            .action(clap::ArgAction::Set)
            .requires("compile"))
        .arg(Arg::new("target")
            .long("target")
            .short('t')
            .help(format!("Target triple; defaults to {tt} (for you)"))
            .value_name("TRIPLE")
            .action(clap::ArgAction::Set)
            .requires("compile"))
        .arg(Arg::new("output")
            .long("output")
            .short('o')
            .help("Output file path")
            .requires("compile")
            .value_hint(ValueHint::AnyPath)
            .value_name("FILE")
            .action(clap::ArgAction::Set)
            .requires("compile"))
        .arg(Arg::new("version")
            .short('v')
            .long("version")
            .help("Print version")
            .action(clap::ArgAction::Version))
        .get_matches();

    if let Some(item) = matches.get_one::<String>("compile") {
        let mut output = matches.get_one::<&str>("output").unwrap_or(&"output").to_string();
        let output_type: OutputType = matches.get_one::<String>("out-type").into();
        if !matches.contains_id("output") {output = format!("{output}.{output_type}")}
        let module_id = matches.get_one::<&str>("module-id").unwrap_or(&"main").to_string();
        let optimization: OptLevel = matches.get_one::<String>("optimization").into();
        let target_triple = 
            matches.get_one::<String>("target").map(|t| TargetTriple::create(t))
                .or_else(|| {Some(TargetMachine::get_default_triple())}).unwrap();
        
        let data = CompileJobData {output, target_triple, optimization, module_id, output_type};
        
        compile(item.to_owned(), data);
    }
}
