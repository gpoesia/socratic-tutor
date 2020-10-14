#lang racket

(require brag/support)
(require br-parser-tools/lex)
(require "terms.rkt")
(require "grammar.rkt")

(define (parse-term t)
  (let ([parse-tree (parse (tokenize (open-input-string t)))])
      (parse-tree-to-term (syntax->datum parse-tree))))

(provide parse-term)

(define (parse-tree-to-term t)
  (match t
         [`(term ,t) (parse-tree-to-term t)]
         [`(predicate ,eq) (parse-tree-to-term eq)]
         [`(expr ,e) (parse-tree-to-term e)]
         [`(paren-expr ,_ ,e ,_) (parse-tree-to-term e)]
         [`(expr_l1 ,e) (parse-tree-to-term e)]
         [`(expr_l2 ,e) (parse-tree-to-term e)]
         [`(expr_l3 ,e) (parse-tree-to-term e)]
         [`(expr_l4 ,e) (parse-tree-to-term e)]
         [`(equality ,e1 ,_ ,e2)
           (Predicate 
             'Eq (list (parse-tree-to-term e1) (parse-tree-to-term e2)))]
         [`(sum ,e1 ,_ ,e2)
           (BinOp op+ (parse-tree-to-term e1) (parse-tree-to-term e2))]
         [`(sub ,e1 ,_ ,e2)
           (BinOp op- (parse-tree-to-term e1) (parse-tree-to-term e2))]
         [`(varcoeff ,e1 ,e2)
           (BinOp op* (parse-tree-to-term e1) (parse-tree-to-term e2))]
         [`(prod ,e1 ,_ ,e2)
           (BinOp op* (parse-tree-to-term e1) (parse-tree-to-term e2))]
         [`(prod ,e1 ,e2)
           (BinOp op* (parse-tree-to-term e1) (parse-tree-to-term e2))]
         [`(div ,e1 ,_ ,e2)
           (BinOp op/ (parse-tree-to-term e1) (parse-tree-to-term e2))]
         [`(number ,n)
           (Number n)]
         [`(any_number ,_)
           (AnyNumber)]
         [`(number ,_ ,n)
           (Number (- n))]
         [`(variable ,v)
           (Variable v)]))

(define (tokenize ip)
  (port-count-lines! ip)
  (let* ([expr-lexer
           (lexer-src-pos
             ["(" (token 'LEFT_PAREN)]
             [")" (token 'RIGHT_PAREN)]
             ["=" (token 'REL_EQ)]
             ["+" (token 'OP_PLUS)]
             [(concatenation "-" (repetition 1 +inf.0 numeric))
              (token 'INTEGER (string->number lexeme))]
             ["-" (token 'OP_MINUS)]
             ["*" (token 'OP_TIMES)]
             ["/" (token 'OP_DIV)]
             ["?" (token 'ANY_NUMBER)]
             [(repetition 1 +inf.0 numeric)
              (token 'INTEGER (string->number lexeme))]
             [(repetition 1 +inf.0 alphabetic)
              (token 'VARIABLE lexeme)]
             [whitespace
              (token 'WHITESPACE lexeme #:skip? #t)]
             )]
         [next-token (lambda () (expr-lexer ip))])
    next-token))
