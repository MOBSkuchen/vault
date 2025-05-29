use std::env::temp_dir;
use std::fmt::{Display, Formatter};
use clap::{Arg, ValueHint};
use clap_builder::Command;
use inkwell::context::Context;
use inkwell::targets::{TargetMachine, TargetTriple};
use lld_rx::LldFlavor;
use crate::codegen::Codegen;
use crate::comp_errors::{CodeError, CompilerError};
use crate::compiler::Compiler;
use crate::directives::CompilationConfig;
use crate::filemanager::FileManager;
use crate::lexer::tokenize;
use crate::linker::{lld_link, ProdType};
use crate::parser::Parser;
use crate::utils::{dedup};

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
mod linker;
mod directives;
mod utils;

#[derive(Copy, Clone)]
enum DevDebugLevel {
    Null = 0,       // Nothing
    Regular = 1,    // View finished LLVM-IR
    More = 2,       // Also view AST and tokens
    Full = 3        // Also view LLVM optimizer output
}

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

enum CompOutputType {
    Object,
    Asm,
    IR,
    BC
}

impl CompOutputType {
    pub fn to_f_ext(&self) -> String {
        match self {
            CompOutputType::Object => "o",
            CompOutputType::Asm => "asm",
            CompOutputType::IR => "ir",
            CompOutputType::BC => "bc"
        }.to_string()
    }
}

impl Display for CompOutputType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            CompOutputType::Object => "object",
            CompOutputType::Asm => "asm",
            CompOutputType::IR => "ir",
            &CompOutputType::BC => "bc"
        })
    }
}

impl From<Option<&String>> for CompOutputType {
    fn from(value: Option<&String>) -> Self {
        match value {
            None => { CompOutputType::Object}
            Some(s) => {
                let s = s.to_lowercase();
                if s == "object" || s == "o" || s == "obj" { CompOutputType::Object}
                else if s == "asm" || s == "assembly" { CompOutputType::Asm}
                else if s == "ir" || s == "ll" { CompOutputType::IR}
                else if s == "bc" || s == "bitcode" { CompOutputType::BC}
                else {println!("unrecognized output-type `{s}`, defaulting to object"); CompOutputType::Object}
            }
        }
    }
}

enum LinkOutputType {
    Coff,
    Elf,
    Wasm,
    MachO,
}

impl Into<LldFlavor> for LinkOutputType {
    fn into(self) -> LldFlavor {
        match self {
            LinkOutputType::Coff => LldFlavor::Coff,
            LinkOutputType::Elf => LldFlavor::Elf,
            LinkOutputType::Wasm => LldFlavor::Wasm,
            LinkOutputType::MachO => LldFlavor::MachO
        }
    }
}

impl LinkOutputType {
    pub fn to_f_ext(&self, lib: bool) -> String {
        match self {
            LinkOutputType::Coff => { if lib { ".dll" } else { ".exe" } },
            LinkOutputType::Elf => "",
            LinkOutputType::Wasm => ".wasm",
            LinkOutputType::MachO => ""
        }.to_string()
    }
}

impl Display for LinkOutputType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            LinkOutputType::Coff => "COFF",
            LinkOutputType::Elf => "Elf",
            LinkOutputType::Wasm => "Wasm",
            LinkOutputType::MachO => "MachO"
        })
    }
}

impl From<Option<&String>> for LinkOutputType {
    fn from(value: Option<&String>) -> Self {
        match value {
            // TODO: This expects to run on windows
            None => { LinkOutputType::Coff}
            Some(s) => {
                let s = s.to_lowercase();
                if s == "coff" || s == "exe" || s == "dll" || s == "win" { LinkOutputType::Coff}
                else if s == "elf" || s == "linux" { LinkOutputType::Elf}
                else if s == "wasm" || s == "web" { LinkOutputType::Wasm}
                else if s == "macho" || s == "mac" { LinkOutputType::MachO}
                else {println!("unrecognized output-type `{s}`, defaulting to object"); LinkOutputType::Coff}
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
    output_type: CompOutputType,
    dev_debug_level: DevDebugLevel,
    debug: bool,
}

struct LinkJobData {
    output: String,
    output_type: LinkOutputType,
    lib: bool,
    libs: Vec<String>,
    stdlib: bool,
    entry: String,
    dev_debug_level: DevDebugLevel,
    debug: bool,
}

impl Display for CompileJobData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Module `{}` for `{}` to output file: `{}` (type: {}) with optimization {}", 
               self.module_id, self.target_triple, 
               self.output, self.output_type, 
               self.optimization)
    }
}

#[derive(Debug)]
enum MixedError {
    CompilerError(CompilerError),
    CodeError(CodeError)
}

impl From<CodeError> for MixedError {
    fn from(value: CodeError) -> Self {
        Self::CodeError(value)
    }
}

impl From<CompilerError> for MixedError {
    fn from(value: CompilerError) -> Self {
        Self::CompilerError(value)
    }
}

pub type MixedResult<T> = Result<T, MixedError>;

fn compile_job(file_manager: &FileManager, compile_job_data: CompileJobData) -> MixedResult<CompilationConfig> {
    let tokens = tokenize(file_manager.get_content())?;
    println!("Compiling `{}` with profile:\n{compile_job_data}", file_manager.input_file);
    
    if compile_job_data.dev_debug_level as u32 >= 2 { 
        println!("Parsed Tokens:\n{:#?}", tokens);
    }
    
    let parser = Parser::new(tokens, file_manager);
    let ast = parser.parse(&mut 0)?;

    if compile_job_data.dev_debug_level as u32 >= 2 {
        println!("Parsed AST:\n{:#?}", ast);
    }

    let context = Context::create();
    let builder = context.create_builder();
    let module = context.create_module(&compile_job_data.module_id);
    let compiler = Compiler::new(&context, &builder, compile_job_data.module_id, file_manager);

    let mut compilation_config = CompilationConfig::new(compile_job_data.debug);
    let module = compiler.compile(module, ast, &mut compilation_config, file_manager)?;
    if compile_job_data.dev_debug_level as u32 >= 1 {
        println!("LLVM-Module (pre optimize):");
        module.print_to_stderr();
    }
    match module.verify() {
        Ok(_) => {}
        Err(msg) => {
            Err(CompilerError::VerifyFailed(msg.to_string()))?
        }
    }
    let codegen = Codegen::new(Some(compile_job_data.target_triple), None, None, None);
    codegen.optimize(&module, &compile_job_data.optimization, compile_job_data.dev_debug_level);
    if compile_job_data.dev_debug_level as u32 >= 1 {
        println!("LLVM-Module (post optimize):");
        module.print_to_stderr();
    }
    let file = match compile_job_data.output_type {
        CompOutputType::Object => codegen.gen_obj(&module, compile_job_data.output),
        CompOutputType::Asm => codegen.gen_asm(&module, compile_job_data.output),
        CompOutputType::IR => codegen.gen_ir(&module, compile_job_data.output),
        CompOutputType::BC => codegen.gen_bc(&module, compile_job_data.output)
    };
    println!("Finished writing to `{file}`!");
    Ok(compilation_config)
}

fn compile(filepath: String, compile_job_data: CompileJobData) {
    let file_manager_r = FileManager::new_from(filepath);
    if let Err(item) = file_manager_r {
        item.output();
        return;
    }

    let file_manager = file_manager_r.unwrap();

    let x = compile_job(&file_manager, compile_job_data);
    if let Err(item) = x {
        match item {
            MixedError::CompilerError(comp_err) => comp_err.output(),
            MixedError::CodeError(code_err) => code_err.visualize_error(&file_manager),
        }
        
        println!("\nAn error has occurred during compilation, terminating compilation.")
    }
    
    // TODO: Handle compilation config result
}

fn compile_and_link(filepath: String, link_job_data: LinkJobData) {
    let file_manager_r = FileManager::new_from(filepath);
    if let Err(item) = file_manager_r {
        item.output();
        return
    }

    let file_manager = file_manager_r.unwrap();

    let mut tmp_file = temp_dir();
    tmp_file.push(format!("tmp_{}.o", link_job_data.output));
    let tmp_file = tmp_file.to_str().unwrap().to_string();

    let compile_job_data = CompileJobData {
        output: tmp_file.clone(),
        target_triple: TargetMachine::get_default_triple(),
        optimization: OptLevel::Full,
        module_id: "main".to_string(),
        output_type: CompOutputType::Object,
        dev_debug_level: link_job_data.dev_debug_level,
        debug: link_job_data.debug
    };

    let x = compile_job(&file_manager, compile_job_data);
    if let Err(item) = x {
        match item {
            MixedError::CompilerError(comp_err) => comp_err.output(),
            MixedError::CodeError(code_err) => code_err.visualize_error(&file_manager),
        }
        println!("\nAn error has occurred during compilation, terminating compilation.");
        return
    }
    
    let mut compilation_config = x.unwrap();

    let mut libs = link_job_data.libs;
    libs.append(&mut compilation_config.libs);
    // Required libs for windows
    if link_job_data.stdlib {
        libs.push("vault-stdlib-win.lib".to_string());
        libs.push("kernel32.lib".to_string());
        libs.push("ucrt.lib".to_string());
        libs.push("msvcrt.lib".to_string());
    }
    
    // Remove duplicates from libs
    dedup(&mut libs);

    println!("Linking `{tmp_file}` with {} libs: {}", libs.len(), libs.iter().map(|x1| {format!("`{x1}`")}).collect::<Vec<String>>().join(", "));

    let linker_result = lld_link(link_job_data.output_type.into(),
             vec![tmp_file], &link_job_data.output, link_job_data.lib, libs,
             if link_job_data.lib { None } else { Some(link_job_data.entry) }, ProdType::Console);
    
    match linker_result {
        Ok(_) => {
            println!("Finished writing to `{}`", link_job_data.output);
        }
        Err(msg) => {
            print!("Failed to link, terminating building. Reason:\n{msg}");
        }
    }
}

fn main() {
    let tt = TargetMachine::get_default_triple();
    let tt = tt.as_str().to_str().unwrap();
    let matches = Command::new(NAME)
        .about(DESCRIPTION)
        .version(VERSION)
        .author("MOBSkuchen")
        .help_template("
{before-help}{name} {version} \"{about}\"
by {author} -> Powered by LLVM
{usage-heading} {usage}

{all-args}{after-help}

Available output types: ASM, IR, BC, OBJECT (default)

This is a temporary build - critical breaking changes WILL occur. Be warned.
")
        .color(clap_builder::ColorChoice::Never)
        .disable_version_flag(true)
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("compile")
                .about("Compile a file")
                .arg(
                    Arg::new("file")
                        .help("File to compile")
                        .value_hint(ValueHint::AnyPath)
                        .value_name("FILE")
                        .required(true),
                )
                .arg(
                    Arg::new("out-type")
                        .long("output-type")
                        .visible_alias("ot")
                        .help("Set the output type instead of inferring it [FOR COMPILING]")
                        .value_name("TYPE")
                        .required(false),
                )
                .arg(
                    Arg::new("optimization")
                        .long("optimization")
                        .short('O')
                        .help("Set optimization level 0-3")
                        .value_name("LEVEL"),
                )
                .arg(
                    Arg::new("module-id")
                        .long("module-id")
                        .help("LLVM module id")
                        .value_name("ID"),
                )
                .arg(
                    Arg::new("target")
                        .long("target")
                        .short('t')
                        .help(format!("Target triple; defaults to {tt} (for you)"))
                        .value_name("TRIPLE"),
                )
                .arg(
                    Arg::new("output")
                        .long("output")
                        .short('o')
                        .help("Output file path")
                        .value_hint(ValueHint::AnyPath)
                        .value_name("FILE"),
                )
                .arg(
                    Arg::new("debug")
                        .long("debug")
                        .short('d')
                        .help("Enable debug mode")
                        .action(clap::ArgAction::SetTrue),
                )
        )
        .subcommand(
            Command::new("build")
                .about("Build a file")
                .arg(
                    Arg::new("file")
                        .help("File to build")
                        .value_hint(ValueHint::AnyPath)
                        .value_name("FILE")
                        .required(true),
                )
                .arg(
                    Arg::new("no-standard")
                        .long("no-standard-lib")
                        .help("Do not link in standard library")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("libs")
                        .short('L')
                        .long("libs")
                        .help("Link in libraries")
                        .value_delimiter(',')
                        .value_hint(ValueHint::AnyPath)
                        .value_name("LIBS"),
                )
                .arg(
                    Arg::new("entry")
                        .long("entry-symbol")
                        .help("Set entry for linking")
                        .value_name("SYMBOL"),
                )
                .arg(
                    Arg::new("library")
                        .long("dst-lib")
                        .help("Set linking output type to library")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("debug")
                        .long("debug")
                        .short('d')
                        .help("Enable debug mode")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("out-type")
                        .long("output-type")
                        .visible_alias("ot")
                        .help("Set the output type instead of inferring it [FOR LINKING]")
                        .value_name("TYPE"),
                )
                .arg(
                    Arg::new("output")
                        .long("output")
                        .short('o')
                        .help("Output file path")
                        .value_hint(ValueHint::AnyPath)
                        .value_name("FILE"),
                )
        )
        .arg(
            Arg::new("version")
                .short('v')
                .long("version")
                .help("Print version")
                .action(clap::ArgAction::Version),
        )
        .arg(
            Arg::new("dev-debug")
                .short('Ö')
                .long("dev-debug-level")
                .help("Compiler developer debug printing level")
                .action(clap::ArgAction::Set),
        )
        .get_matches();
    
    let dev_debug_level = if let Some(dev_debug_level) = matches.get_one::<String>("dev-debug") {
        match dev_debug_level.parse().unwrap() {
            0 => DevDebugLevel::Null,
            1 => DevDebugLevel::Regular,
            2 => DevDebugLevel::More,
            3 => DevDebugLevel::Full,
            _ => {
                println!("unknown dev debug level, defaulting to 0");
                DevDebugLevel::Null
            }
        }
    } else {DevDebugLevel::Null};

    match matches.subcommand() {
        Some(("compile", sub)) => {
            let item = sub.get_one::<String>("file").unwrap();
            let mut output = sub.get_one::<&str>("output").unwrap_or(&"output").to_string();
            let output_type: CompOutputType = sub.get_one::<String>("out-type").into();
            if !sub.contains_id("output") {
                output = format!("{output}.{}", output_type.to_f_ext());
            }
            let module_id = sub.get_one::<&str>("module-id").unwrap_or(&"main").to_string();
            let optimization: OptLevel = sub.get_one::<String>("optimization").into();
            let target_triple = sub
                .get_one::<String>("target")
                .map(|t| TargetTriple::create(t))
                .or_else(|| Some(TargetMachine::get_default_triple()))
                .unwrap();

            let data = CompileJobData {
                output,
                target_triple,
                optimization,
                module_id,
                output_type,
                dev_debug_level,
                debug: sub.get_flag("debug")
            };

            compile(item.to_owned(), data);
        }
        Some(("build", sub)) => {
            let item = sub.get_one::<String>("file").unwrap();
            let lib = sub.get_flag("library");
            let no_std = sub.get_flag("no-standard");
            let mut output = sub.get_one::<&str>("output").unwrap_or(&"output").to_string();
            let output_type: LinkOutputType = sub.get_one::<String>("out-type").into();
            if !sub.contains_id("output") {
                output = format!("{output}{}", output_type.to_f_ext(lib));
            }
            let entry = sub.get_one::<&str>("entry").unwrap_or(&"main").to_string();
            let libs: Vec<String> = if let Some(lbs) = sub.get_many("libs") {
                lbs.map(|s: &String| s.to_string()).collect()
            } else {
                vec![]
            };

            let data = LinkJobData {
                output,
                output_type,
                lib,
                stdlib: !no_std,
                libs,
                entry,
                dev_debug_level,
                debug: sub.get_flag("debug")
            };

            compile_and_link(item.to_owned(), data);
        }
        _ => unreachable!(),
    }
}
