#lang brag

; This is a unified grammar for all domains we support.
; A term is the union of terms in all domains. Each domain is defined below.

term       : Dequations | Dternary | Dsorting | Dcounting

; ===========================================
; =========== Equations Domain ==============
; ===========================================

Dequations : predicate | expr
predicate  : equality
equality   : expr REL_EQ expr
expr       : expr_l1

expr_l1    : sum | sub | expr_l2
sum        : expr_l1 OP_PLUS expr_l2
sub        : expr_l1 OP_MINUS expr_l2

expr_l2    : prod | div | expr_l3
prod       : expr_l2 OP_TIMES expr_l3
             | paren-expr paren-expr
varcoeff   : number expr_l4 | neg_number expr_l4
div        : expr_l2 OP_DIV expr_l3

expr_l3    : number | neg_number | any_number | varcoeff | expr_l4
expr_l4    : variable | neg_var | paren-expr
paren-expr : LEFT_PAREN expr RIGHT_PAREN
variable   : VARIABLE
neg_var    : OP_MINUS VARIABLE
any_number : ANY_NUMBER
neg_number : OP_MINUS INTEGER
number     : INTEGER

; ===========================================
; =========== Ternary Domain ================
; ===========================================

Dternary       : ternary_number
ternary_number : TERNARY_MARK LEFT_PAREN ternary_digits
ternary_digits : ternary_end | ternary_cons
ternary_cons   : ternary_digit ternary_digits
ternary_digit  : VARIABLE INTEGER
ternary_end    : RIGHT_PAREN

; ===========================================
; =========== Sorting domain ================
; ===========================================

Dsorting       : sorting_list
sorting_list   : sorting_single | sorting_many
sorting_single : SORTING_ELEM
sorting_many   : SORTING_ELEM SORTING_SEP sorting_list

; ============================================
; =========== Counting domain ================
; ============================================

Dcounting : INTEGER COMMA INTEGER COMMA ELLIPSIS
