use crate::comp_errors::{CodeError, CodeResult};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::Range;

#[derive(PartialEq, Copy, Debug, Clone, Hash)]
pub enum TokenType {
    // Keywords
    Function,
    Export,
    Import,
    Extern,

    Identifier,
    String,
    NumberInt,
    NumberFloat,

    I32,
    I64,
    F32,
    F64,
    Void,
    U8,
    U32,
    U64,

    LParen,
    RParen,
    Comma,
    Dot,
    Plus,
    Minus,
    Slash,
    Star,
    Colon,
    SemiColon,
    Greater,
    Lesser,
    Pipe,
    And,
    Exclamation,
    Equals,
    DoubleEquals,
    NotEquals,
    GreaterEquals,
    LesserEquals,
    RBrace,
    LBrace,
    As,
    Ref,
    Private,
    Return,
    If,
    Else,
    Elif,
    For,
    While,
    Let,
    Or,
    Ptr,
    Break,
    Continue,
    Bool,
    Struct,
    New,
    Relative,
    Malloc,
    Del,
    Directive,
    ModuleAccess,
    LBrackets,
    RBrackets,
    QuestionMark,

    // Virtual types
    Expression,
    Statement,
}

impl TokenType {
    pub fn visualize(&self) -> String {
        (match self {
            TokenType::Function => "fun",
            TokenType::Export => "export",
            TokenType::Import => "import",
            TokenType::Extern => "extern",
            TokenType::Identifier => "Identifier",
            TokenType::String => "String",
            TokenType::NumberInt => "Integer",
            TokenType::NumberFloat => "Floating-point",
            TokenType::LParen => "(",
            TokenType::RParen => ")",
            TokenType::Comma => ",",
            TokenType::Dot => ".",
            TokenType::Plus => "+",
            TokenType::Minus => "-",
            TokenType::Slash => "/",
            TokenType::Star => "*",
            TokenType::Colon => ":",
            TokenType::SemiColon => ";",
            TokenType::Greater => ">",
            TokenType::Lesser => "<",
            TokenType::Pipe => "|",
            TokenType::And => "&&",
            TokenType::Ref => "&",
            TokenType::Exclamation => "!",
            TokenType::Equals => "=",
            TokenType::DoubleEquals => "==",
            TokenType::NotEquals => "!=",
            TokenType::GreaterEquals => ">=",
            TokenType::LesserEquals => "<=",
            TokenType::RBrace => "}",
            TokenType::LBrace => "{",
            TokenType::As => "=>",
            TokenType::Private => "private",
            TokenType::Return => "return",
            TokenType::Expression => "Expression",
            TokenType::Statement => "Statement",
            TokenType::For => "for",
            TokenType::If => "if",
            TokenType::Else => "else",
            TokenType::Elif => "elif",
            TokenType::Let => "let",
            TokenType::New => "new",
            TokenType::While => "while",
            TokenType::Break => "break",
            TokenType::Continue => "continue",
            TokenType::Bool => "bool",
            TokenType::I32 => "i32",
            TokenType::F32 => "f32",
            TokenType::Void => "void",
            TokenType::Or => "or",
            TokenType::Ptr => "ptr",
            TokenType::I64 => "i64",
            TokenType::F64 => "f64",
            TokenType::U8 => "u8",
            TokenType::U32 => "u32",
            TokenType::U64 => "u64",
            TokenType::Struct => "struct",
            TokenType::Relative => "~",
            TokenType::Malloc => "|>",
            TokenType::Del => "del",
            TokenType::Directive => "#",
            TokenType::ModuleAccess => "::",
            TokenType::LBrackets => "[",
            TokenType::RBrackets => "]",
            TokenType::QuestionMark => "?"
        })
            .to_string()
    }
}

impl Display for TokenType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.visualize())
    }
}

#[derive(Debug, Clone, Copy, Hash)]
#[derive(PartialEq)]
pub struct CodePosition {
    pub idx_start: usize,
    pub idx_end: usize,
    pub line_start: usize,
    pub line_end: usize,
    pub line_idx_start: usize,
    pub line_idx_end: usize,
}

impl CodePosition {
    pub fn one_char(idx: usize, line: usize, line_idx: usize) -> Self {
        CodePosition {
            idx_start: idx,
            idx_end: idx,
            line_start: line,
            line_end: line,
            line_idx_start: line_idx - 1,
            line_idx_end: line_idx,
        }
    }

    pub fn encode(&self) -> String {
        format!(
            "{},{},{},{},{},{}",
            self.idx_start,
            self.idx_end,
            self.line_start,
            self.line_end,
            self.line_idx_start,
            self.line_idx_end
        )
    }

    // MUST BE A VALID STRING!!!
    pub fn decode(s: &str) -> Self {
        println!("{}", s);
        let parts: Vec<&str> = s.split(',').collect();

        if parts.len() != 6 {
            return ".".parse::<usize>().map(|_| unreachable!()).unwrap();
        }

        let idx_start = parts[0].parse::<usize>().unwrap();
        let idx_end = parts[1].parse::<usize>().unwrap();
        let line_start = parts[2].parse::<usize>().unwrap();
        let line_end = parts[3].parse::<usize>().unwrap();
        let line_idx_start = parts[4].parse::<usize>().unwrap();
        let line_idx_end = parts[5].parse::<usize>().unwrap();

        CodePosition {
            idx_start,
            idx_end,
            line_start,
            line_end,
            line_idx_start,
            line_idx_end,
        }
    }

    pub fn eof() -> Self {
        CodePosition {
            idx_start: 0,
            line_start: 0,
            idx_end: 0,
            line_end: 0,
            line_idx_start: 0,
            line_idx_end: 0,
        }
    }

    pub fn merge(&self, other: Self) -> Self {
        Self {
            idx_start: self.idx_start,
            idx_end: other.idx_end,
            line_start: self.line_start,
            line_end: other.line_end,
            line_idx_start: self.line_idx_start,
            line_idx_end: other.line_idx_end,
        }
    }

    pub fn inc(&self) -> Self {
        Self {
            idx_start: self.idx_start + 1,
            idx_end: self.idx_end + 1,
            line_start: self.line_start,
            line_end: self.line_end,
            line_idx_start: self.line_idx_start + 1,
            line_idx_end: self.line_idx_end + 1,
        }
    }
}

impl CodePosition {
    pub fn range(&self, offset: usize) -> Range<usize> {
        self.line_idx_start + offset..self.line_idx_end + offset
    }
}

#[derive(Debug, Hash, PartialEq)]
pub struct Token {
    pub content: String,
    pub token_type: TokenType,
    pub code_position: CodePosition,
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}

impl Token {
    pub fn from_one(
        idx: usize,
        line: usize,
        line_idx: usize,
        content: char,
        token_type: TokenType,
    ) -> Self {
        Self {
            content: content.to_string(),
            token_type,
            code_position: CodePosition::one_char(idx, line, line_idx),
        }
    }
}

pub struct Scanner {
    pub cursor: usize,
    pub line: usize,
    pub line_idx: usize,
    pub characters: Vec<char>,
}

impl Scanner {
    pub fn new(string: &str) -> Self {
        Self {
            cursor: 0,
            line: 0,
            line_idx: 0,
            characters: string.chars().collect(),
        }
    }

    /// Returns the next character without advancing the cursor.
    /// AKA "lookahead"
    pub fn peek(&self) -> Option<&char> {
        self.characters.get(self.cursor)
    }

    /// Returns true if further progress is not possible.
    pub fn is_done(&self) -> bool {
        self.cursor == self.characters.len()
    }

    /// Returns the next character (if available) and advances the cursor.
    pub fn pop(&mut self) -> Option<&char> {
        match self.characters.get(self.cursor) {
            Some(character) => {
                self.cursor += 1;
                self.line_idx += 1;
                if *character == '\n' {
                    self.line += 1;
                    self.line_idx = 0;
                }

                Some(character)
            }
            None => None,
        }
    }

    pub fn previous(&self) -> Option<&char> {
        match self.characters.get(self.cursor - 1) {
            Some(character) => Some(character),
            None => None,
        }
    }

    pub fn this_as_token(&self, token_type: TokenType) -> Option<Token> {
        self.previous().map(|c| Token::from_one(
                self.cursor,
                self.line,
                self.line_idx,
                *c,
                token_type,
            ))
    }

    pub fn this_as_codepos(&self) -> Option<CodePosition> {
        if self.is_done() {
            None
        } else {
            Some(CodePosition::one_char(
                self.cursor,
                self.line,
                self.line_idx,
            ))
        }
    }

    pub fn this_as_codepos2(&self) -> CodePosition {
        self.this_as_codepos()
            .expect("This should not happen -> constructing code pos")
    }
}

fn tokenizer(scanner: &mut Scanner) -> CodeResult<Option<Token>> {
    while let Some(current) = scanner.peek() {
        match current {
            ' ' | '\t' | '\n' | '\r' => {
                scanner.pop();
            }

            '/' => {
                scanner.pop(); // Consume the first '/'
                let peek = scanner.peek();
                if peek.is_none() {
                    return Ok(scanner.this_as_token(TokenType::Slash))
                }
                let peek_u = peek.unwrap();
                if *peek_u == '*' {
                    // Multi-line comment
                    scanner.pop(); // Consume the '*'
                    while let Some(c) = scanner.pop() {
                        if *c == '*' && scanner.peek().is_some_and(|t| *t == '/') {
                            scanner.pop(); // Consume the closing '/'
                            break; // Exit the comment loop
                        }
                    }
                } else if *peek_u == '/' {
                    scanner.pop(); // Consume the second '/'
                    // Single-line comment
                    while let Some(c) = scanner.pop() {
                        if *c == '\n' {
                            break; // Exit the comment loop
                        }
                    }
                } else {
                    return Ok(scanner.this_as_token(TokenType::Slash))
                }
            }

            '(' | ')' | ',' | '.' | '+' | '*' | ';' | '{' | '}' | '#' | '[' | ']' | '?' => {
                let token_type = match current {
                    '(' => TokenType::LParen,
                    ')' => TokenType::RParen,
                    ',' => TokenType::Comma,
                    '.' => TokenType::Dot,
                    '+' => TokenType::Plus,
                    '*' => TokenType::Star,
                    '|' => TokenType::Pipe,
                    ';' => TokenType::SemiColon,
                    '{' => TokenType::LBrace,
                    '}' => TokenType::RBrace,
                    '[' => TokenType::LBrackets,
                    ']' => TokenType::RBrackets,
                    '#' => TokenType::Directive,
                    '?' => TokenType::QuestionMark,
                    _ => unreachable!(),
                };
                scanner.pop();
                return Ok(scanner.this_as_token(token_type));
            }
            '&' => {
                scanner.pop();
                if let Some('&') = scanner.peek() {
                    scanner.pop();
                    return Ok(scanner.this_as_token(TokenType::And));
                }
                return Ok(scanner.this_as_token(TokenType::Ref));
            }
            ':' => {
                scanner.pop();
                if let Some(':') = scanner.peek() {
                    scanner.pop();
                    return Ok(scanner.this_as_token(TokenType::ModuleAccess));
                }
                return Ok(scanner.this_as_token(TokenType::Colon));
            }
            '-' => {
                // Record the start position *before* consuming the '-' so we can slice
                let start_pos = scanner.cursor;
                scanner.pop(); // consume the '-'

                // If the next character is a digit, parse a (possibly floating-point) literal
                if let Some(next) = scanner.peek() {
                    if next.is_digit(10) {
                        let mut is_float = false;
                        // consume all digits, and at most one '.'
                        while let Some(&c) = scanner.peek() {
                            if c.is_digit(10) {
                                scanner.pop();
                            } else if c == '.' && !is_float {
                                is_float = true;
                                scanner.pop();
                            } else {
                                break;
                            }
                        }

                        // collect the slice “-123.45”
                        let number: String = scanner.characters[start_pos..scanner.cursor]
                            .iter()
                            .collect();
                        let token_type = if is_float {
                            TokenType::NumberFloat
                        } else {
                            TokenType::NumberInt
                        };
                        return Ok(Some(Token {
                            content: number.clone(),
                            token_type,
                            code_position: CodePosition {
                                idx_start: start_pos,
                                idx_end: scanner.cursor,
                                line_start: scanner.line,
                                line_end: scanner.line,
                                // line_idx_start should back up by the full literal length
                                line_idx_start: scanner.line_idx - number.len(),
                                line_idx_end: scanner.line_idx,
                            },
                        }));
                    }
                }

                // Otherwise it was just a minus operator
                return Ok(scanner.this_as_token(TokenType::Minus));
            }
            '~' => {
                scanner.pop();
                return Ok(scanner.this_as_token(TokenType::Relative));
            }
            '>' => {
                scanner.pop();
                if let Some('=') = scanner.peek() {
                    scanner.pop();
                    return Ok(scanner.this_as_token(TokenType::GreaterEquals));
                }
                return Ok(scanner.this_as_token(TokenType::Greater));
            }
            '<' => {
                scanner.pop();
                if let Some('=') = scanner.peek() {
                    scanner.pop();
                    return Ok(scanner.this_as_token(TokenType::LesserEquals));
                }
                return Ok(scanner.this_as_token(TokenType::Lesser));
            }
            '!' => {
                scanner.pop();
                if let Some('=') = scanner.peek() {
                    scanner.pop();
                    return Ok(scanner.this_as_token(TokenType::NotEquals));
                }
                return Ok(scanner.this_as_token(TokenType::Exclamation));
            }
            '=' => {
                scanner.pop();
                if let Some('=') = scanner.peek() {
                    scanner.pop();
                    return Ok(scanner.this_as_token(TokenType::DoubleEquals));
                } else if let Some('>') = scanner.peek() {
                    scanner.pop();
                    return Ok(scanner.this_as_token(TokenType::As));
                }
                return Ok(scanner.this_as_token(TokenType::Equals));
            }
            '|' => {
                scanner.pop();
                if let Some('>') = scanner.peek() {
                    scanner.pop();
                    return Ok(scanner.this_as_token(TokenType::Malloc));
                }
                return Ok(scanner.this_as_token(TokenType::Pipe));
            }

            // Identifiers and keywords
            c if c.is_alphabetic() || *c == '_' => {
                let start_pos = scanner.cursor;
                while let Some(next) = scanner.peek() {
                    if next.is_alphanumeric() || *next == '_' {
                        scanner.pop();
                    } else {
                        break;
                    }
                }
                let identifier: String = scanner.characters[start_pos..scanner.cursor]
                    .iter()
                    .collect();
                let token_type = match identifier.as_str() {
                    "fun" => TokenType::Function,
                    "export" => TokenType::Export,
                    "import" => TokenType::Import,
                    "extern" => TokenType::Extern,
                    "let" => TokenType::Let,
                    "for" => TokenType::For,
                    "private" => TokenType::Private,
                    "return" => TokenType::Return,
                    "break" => TokenType::Break,
                    "while" => TokenType::While,
                    "continue" => TokenType::Continue,
                    "struct" => TokenType::Struct,
                    "del" => TokenType::Del,
                    "i32" => TokenType::I32,
                    "f32" => TokenType::F32,
                    "f64" => TokenType::F64,
                    "i64" => TokenType::I64,
                    "u32" => TokenType::U32,
                    "u64" => TokenType::U64,
                    "u8" => TokenType::U8,
                    "void" => TokenType::Void,
                    "ptr" => TokenType::Ptr,
                    "new" => TokenType::New,
                    "if" => TokenType::If,
                    "elif" => TokenType::Elif,
                    "else" => TokenType::Else,
                    _ => TokenType::Identifier,
                };
                return Ok(Some(Token {
                    content: identifier.clone(),
                    token_type,
                    code_position: CodePosition {
                        idx_start: start_pos,
                        idx_end: scanner.cursor,
                        line_start: scanner.line,
                        line_end: scanner.line,
                        line_idx_start: scanner.line_idx - identifier.len(),
                        line_idx_end: scanner.line_idx,
                    },
                }));
            }

            // Numbers
            c if c.is_digit(10) => {
                let start_pos = scanner.cursor;
                let mut is_float = false;
                while let Some(next) = scanner.peek() {
                    if next.is_digit(10) {
                        scanner.pop();
                    } else if *next == '.' && !is_float {
                        is_float = true;
                        scanner.pop();
                    } else {
                        break;
                    }
                }
                let number: String = scanner.characters[start_pos..scanner.cursor]
                    .iter()
                    .collect();
                let token_type = if is_float {
                    TokenType::NumberFloat
                } else {
                    TokenType::NumberInt
                };
                return Ok(Some(Token {
                    content: number.clone(),
                    token_type,
                    code_position: CodePosition {
                        idx_start: start_pos,
                        idx_end: scanner.cursor,
                        line_start: scanner.line,
                        line_end: scanner.line,
                        line_idx_start: scanner.line_idx - number.len(),
                        line_idx_end: scanner.line_idx,
                    },
                }));
            }

            // Strings
            '"' => {
                scanner.pop(); // Consume opening quote
                let start_pos = scanner.cursor;
                while let Some(next) = scanner.peek() {
                    if *next == '"' {
                        let string: String = scanner.characters[start_pos..scanner.cursor]
                            .iter()
                            .collect();
                        scanner.pop(); // Consume closing quote
                        return Ok(Some(Token {
                            content: string.clone(),
                            token_type: TokenType::String,
                            code_position: CodePosition {
                                idx_start: start_pos,
                                idx_end: start_pos,
                                line_start: scanner.line,
                                line_end: scanner.line,
                                line_idx_start: scanner.line_idx - string.len() - 2,
                                line_idx_end: scanner.line_idx,
                            },
                        }));
                    } else {
                        scanner.pop();
                    }
                }
                return Err(CodeError::new_eof_error());
            }
            _ => {
                return Err(CodeError::new_unknown_char_error(
                    scanner.this_as_codepos2(),
                    *current,
                ));
            }
        }
    }
    Ok(None)
}

pub fn tokenize(content: String) -> CodeResult<Vec<Token>> {
    let mut scanner = Scanner::new(content.as_str());
    let mut tokens: Vec<Token> = vec![];
    loop {
        let token = tokenizer(&mut scanner)?;
        if let Some(item) = token {
            tokens.push(item)
        } else {
            return Ok(tokens);
        }
    }
}
