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

         ; Equations domain.
         [`(Dequations ,t) (parse-tree-to-term t)]
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
         [`(int_frac ,n ,_ ,m)
           (Number (/ n m))]
         [`(any_number ,_)
           (AnyNumber)]
         [`(neg_number ,_ ,n)
           (Number (- n))]
         [`(neg_frac ,_ ,f)
           (Number (- (Number-value (parse-tree-to-term f))))]
         [`(neg_var ,_ ,v)
           (BinOp op* (Number -1) (Variable v))]
         [`(variable ,v)
           (Variable v)]

         ; Ternary domain.
         [`(Dternary ,t) (parse-tree-to-term t)]
         [`(ternary_number ,_ ,_ ,ds)
          (TernaryNumber (parse-tree-to-term ds))]
         [`(ternary_digits ,ds) (parse-tree-to-term ds)]
         [`(ternary_cons ,d ,ds) (cons (parse-tree-to-term d) (parse-tree-to-term ds))]
         [`(ternary_digit ,v ,i) (TernaryDigit (- (char->integer (car (string->list v)))
                                                  (char->integer #\a))
                                               i)]
         [`(ternary_end ,_) empty]

         ; Counting domain
         [`(Dcounting ,l ,_ ,r ,_ ,_) (CountingSequence l r)]

         ; Sorting domain
         [`(Dsorting ,t) (SortingList (parse-tree-to-term t))]
         [`(sorting_list ,t) (parse-tree-to-term t)]
         [`(sorting_single ,n) (list n)]
         [`(sorting_many ,h ,_ ,t) (cons h (parse-tree-to-term t))]
         ; Fraction domain
        [`(Dfraction ,_ ,t) (parse-tree-to-term t)]
        [`(fexpr_l1 ,t) (parse-tree-to-term t)]
        [`(fsum ,a ,_ ,b) (BinOp op+ (parse-tree-to-term a) (parse-tree-to-term b))]
        [`(fsub ,a ,_ ,b) (BinOp op- (parse-tree-to-term a) (parse-tree-to-term b))]
        [`(fexpr_l2 ,t) (parse-tree-to-term t)]
        [`(fprod ,e1 ,e2)
          (BinOp op* (parse-tree-to-term e1) (parse-tree-to-term e2))]
        [`(fprod ,e1 ,_ ,e2)
          (BinOp op* (parse-tree-to-term e1) (parse-tree-to-term e2))]
        [`(fvarcoeff ,e1 ,e2)
          (BinOp op* (parse-tree-to-term e1) (parse-tree-to-term e2))]
        [`(fraction ,e1 ,_ ,e2)
          (BinOp op/ (parse-tree-to-term e1) (parse-tree-to-term e2))]
        [`(fexpr_l3 ,t) (parse-tree-to-term t)]
        [`(fparen-expr ,_ ,e ,_) (parse-tree-to-term e)]
        [`(fnumber ,t) (parse-tree-to-term t)]))

(define (tokenize ip)
  (port-count-lines! ip)
  (let* ([expr-lexer
           (lexer-src-pos
             ["(" (token 'LEFT_PAREN)]
             [")" (token 'RIGHT_PAREN)]
             ["[" (token 'LEFT_SBRACKET)]
             ["]" (token 'RIGHT_SBRACKET)]
             ["," (token 'COMMA)]
             ["..." (token 'ELLIPSIS)]
             ["=" (token 'REL_EQ)]
             ["+" (token 'OP_PLUS)]
             [(repetition 1 +inf.0 numeric)
              (token 'INTEGER (string->number lexeme))]
             ["-" (token 'OP_MINUS)]
             ["*" (token 'OP_TIMES)]
             ["//" (token 'OP_FRAC)]
             ["/" (token 'OP_DIV)]
             ["?" (token 'ANY_NUMBER)]
             ["#" (token 'TERNARY_MARK)]
             ["|" (token 'SORTING_SEP)]
             ["F" (token 'FRACTION_MARK)]
             [(repetition 1 +inf.0 "_")
              (token 'SORTING_ELEM (string-length lexeme))]
             [(repetition 1 +inf.0 alphabetic)
              (token 'VARIABLE lexeme)]
             [whitespace
              (token 'WHITESPACE lexeme #:skip? #t)])]


         [next-token (lambda () (expr-lexer ip))])
    next-token))
