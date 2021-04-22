#lang brag

; This is a unified grammar for all domains we support.
; A term is the union of terms in all domains. Each domain is defined below.

term       : Dequations | Dternary | Dsorting | Dcounting | Dfraction

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

expr_l3    : number | int_frac | neg_number | neg_frac | any_number | varcoeff | expr_l4
expr_l4    : variable | neg_var | paren-expr
paren-expr : LEFT_PAREN expr RIGHT_PAREN
variable   : VARIABLE
neg_var    : OP_MINUS VARIABLE
any_number : ANY_NUMBER
neg_number : OP_MINUS INTEGER
neg_frac   : OP_MINUS int_frac
number     : INTEGER
int_frac   : INTEGER OP_FRAC INTEGER

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


; ============================================
; =========== Fractions domain ================
; ============================================

; 3/5
; 3/5 + 12/10 + 2/4

Dfraction: FRACTION_MARK fexpr_l1

fexpr_l1    : fsum | fsub | fexpr_l2
fsum        : fexpr_l1 OP_PLUS fexpr_l2
fsub        : fexpr_l1 OP_MINUS fexpr_l2

fexpr_l2    : fprod | fraction | fexpr_l3
fprod       : fexpr_l2 OP_TIMES fexpr_l3
             | fparen-expr fparen-expr
fvarcoeff   : fnumber fparen-expr
fraction    : fexpr_l2 OP_DIV fexpr_l3

fexpr_l3    : fnumber | fvarcoeff | fparen-expr
fparen-expr : LEFT_PAREN fexpr_l1 RIGHT_PAREN

fnumber     : neg_number | number