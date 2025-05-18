use crate::codeviz::print_code_error;
use crate::filemanager::FileManager;
use crate::lexer::{CodePosition, Token, TokenType};
use colorize_rs::{AnsiColor, Color};
use std::fmt;
use std::fmt::format;

#[derive(Debug)]
pub enum CompilerError {
    FileNotAccessible(String, bool),
    FileCorrupted(String),
}

impl CompilerError {
    pub fn output(&self) {
        match self {
            CompilerError::FileNotAccessible(f, s) => {
                println!(
                    "Could not read input file `{}`, because the file is not accessible.",
                    f.to_string().bold().b_yellow().underlined()
                );
                if *s {
                    println!(
                        "{}",
                        "This is because the directory does not exist!".bold().red()
                    );
                }
            }
            CompilerError::FileCorrupted(f) => {
                println!("Could not read input file `{}`, because the file is corrupted or otherwise not understandable.", f.to_string().bold().underlined())
            }
        }
    }
}

#[derive(Debug)]
pub enum CodeErrorType {
    LexerUnknownChar,
    LexerUnexpectedChar,
    LexerEndOfFile,
    ParserUnexpectedToken,
    MissingTokenError,
    FunctionOverloaded,
    NotAType,
}

#[derive(Debug)]
pub enum CodeWarningType {
    DeadCode,
    UnnecessaryCode,
    DiscouragedPractice,
}

#[derive(Debug)]
pub struct CodeError {
    pub position: CodePosition,
    pub code_error_type: CodeErrorType,
    pub title: String,
    pub footer: String,
    pub pointer: Option<String>,
    pub notes: Vec<String>,
}

impl CodeError {
    pub fn new(
        position: CodePosition,
        code_error_type: CodeErrorType,
        title: String,
        pointer: Option<String>,
        footer: String,
        notes: Vec<String>,
    ) -> Self {
        Self {
            position,
            code_error_type,
            title,
            footer,
            pointer,
            notes,
        }
    }

    pub fn placeholder() -> Self {
        panic!("Please remove this placeholder!");
    }

    pub fn new_unexpected_token_error(
        token: &Token,
        expected: TokenType,
        extra: Option<String>,
    ) -> Self {
        Self::new(
            token.code_position,
            CodeErrorType::ParserUnexpectedToken,
            "Unexpected Token".to_string(),
            Some(format!("Expected `{}` here", expected)),
            format!(
                "Expected another token `{}`, but got `{}`",
                expected, token.token_type
            ),
            if extra.is_some() {
                vec![extra.unwrap()]
            } else {
                vec![]
            }
        )
    }

    pub fn new_unknown_char_error(position: CodePosition, c: char) -> Self {
        Self::new(
            position,
            CodeErrorType::LexerUnknownChar,
            "Unknown character".to_string(),
            Some("This one".to_string()),
            format!("Character `{}` is weird!", c),
            vec![],
        )
    }

    pub fn new_eof_error() -> Self {
        Self::new(
            CodePosition::eof(),
            CodeErrorType::LexerEndOfFile,
            "End of File".to_string(),
            None,
            "Premature end of file!".to_string(),
            vec![]
        )
    }

    pub fn missing_token_error(last_token: &Token) -> Self {
        Self::new(
            last_token.code_position,
            CodeErrorType::MissingTokenError,
            "Missing token".to_string(),
            Some("After this".to_string()),
            "Premature end of file!".to_string(),
            vec![],
        )
    }

    pub fn missing_ends_error(code_position: CodePosition, token_type: TokenType) -> Self {
        Self::new(
            code_position,
            CodeErrorType::ParserUnexpectedToken,
            "Wrong token, expected other".to_string(),
            Some(format!("help: add `{}` here (or after)", token_type)),
            format!("Expected a `{}` at end the statement", token_type),
            vec![]
        )
    }

    pub fn not_a_type_error(token: &Token) -> Self {
        Self::new(token.code_position, CodeErrorType::NotAType, "Wrong token, expected a type".to_string(), None, format!("Expected a valid type, not `{token}`"), vec![])
    }

    pub fn function_overloaded(already_token: &Token) -> Self {
        Self::new(
            already_token.code_position,
            CodeErrorType::FunctionOverloaded,
            "Function was overloaded".to_string(),
            Some("Function mode set here".to_string()),
            "Can not have multiple function modes".to_string(),
            vec!["Remove one of the modifiers".to_string()],
        )
    }

    pub fn visualize_error(self, file_manager: &FileManager) {
        print_code_error(self, file_manager)
    }
}

pub type CompResult<T> = Result<T, CompilerError>;
pub type CodeResult<T> = Result<T, CodeError>;

impl fmt::Display for CodeErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct CodeWarning {
    pub position: CodePosition,
    pub code_warn_type: CodeWarningType,
    pub title: String,
    pub footer: String,
    pub pointer: Option<String>,
    pub notes: Vec<String>,
}

impl CodeWarning {
    pub fn new(
        position: CodePosition,
        code_warn_type: CodeWarningType,
        title: String,
        footer: String,
        pointer: Option<String>,
        notes: Vec<String>,
    ) -> Self {
        Self {
            position,
            code_warn_type,
            title,
            footer,
            pointer,
            notes,
        }
    }

    pub fn new_unnecessary_code(position: CodePosition, extra: Option<String>) -> Self {
        Self::new(
            position,
            CodeWarningType::UnnecessaryCode,
            "Unnecessary code".to_string(),
            "This code does not change the outcome".to_string(),
            None,
            if extra.is_some() {
                vec![extra.unwrap()]
            } else {
                vec!["You should remove it".to_string()]
            },
        )
    }
}
