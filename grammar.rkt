#lang brag

term       : predicate | expr

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
