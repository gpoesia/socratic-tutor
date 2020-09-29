#lang algebraic/racket/base
(require algebraic/data)
(require algebraic/function)
(require algebraic/racket/base/forms)
(require racket/match)
(require rebellion/type/enum)
(require "debug.rkt")

(data Term (Number Variable UnOp BinOp AnyNumber Predicate))
(define-enum-type Operator (op+ op- op* op/))

(define Predicate-type (phi (Predicate type _) type))
(define Predicate-terms (phi (Predicate _ terms) terms))

(define (compute-bin-op op a b)
  (match op
    [(== op+) (+ a b)]
    [(== op-) (- a b)]
    [(== op*) (* a b)]
    [(== op/) (/ a b)]
    ))

; Returns the size of the term `t` (also corresponding to the
; number of subterms it has).
(define term-size
  (function
    [(Number n) 1]
    [(Variable v) 1]
    [(AnyNumber) 1]
    [(UnOp op t1) (+ 1 (term-size t1))]
    [(BinOp op t1 t2) (+ 1 (term-size t1) (term-size t2))]
    [(Predicate type terms) (foldl + 1 (map term-size terms))]
    ))

; Returns a list with the direct subterms of `t`.
(define subterms
  (function
    [(UnOp _ t1) (list t1)]
    [(BinOp _ t1 t2) (list t1 t2)]
    [(Predicate _ terms) terms]
    [t '()]))

; Returns a list with all subterms of `t`, found recursively.
(define (enumerate-subterms t)
  (cons t (apply append (map enumerate-subterms (subterms t)))))

; Takes a term `t` and a list of subterms `new-subterms`, and returns a copy of
; `t` where the subterms are replaced by `new-subterms`.
(define replace-subterms
  (function*
    [((UnOp op _) `(,t)) (UnOp op t)]
    [((BinOp op _ _) `(,t1 ,t2)) (BinOp op t1 t2)]
    [((Predicate type _) terms) (Predicate type terms)]
    [(t _) t]))

; Returns a list of all indices of subterms of `t` for which `p` is true.
(define (filter-subterms t p)
  (car (filter-subterms-aux (list) t p 0)))

; Returns (l . index)
(define (filter-subterms-aux l t p index)
  (let ([result (if (p t) (cons index l) l)])
    (foldl
      (lambda (st r)
        (let ([(sts . idx) r])
          (filter-subterms-aux sts st p idx)))
      (cons result (+ 1 index))
      (subterms t))))

; Returns whether a pair of terms matches (i.e. are equal except that
; AnyNumber is considered equal to Number). Does not recurse: only compares
; the top level (which we might want to change at some point).
(define term-matches?
  (function*
    [(x x) #t]
    [((AnyNumber) (Number x)) #t]
    [_ #f]
  ))

(define (goal-matches? a b)
  (if (and (Predicate? a) (Predicate? b))
    (and (eq? (Predicate-type a) (Predicate-type b))
         (andmap goal-matches? (Predicate-terms a) (Predicate-terms b)))
    (term-matches? a b)))

; Locates the sub-term in `term` with index `index` and replaces that sub-term
; with the return value of `transform` applied on it (or leaves it intact if
; `transform` returns #f).
(define (rewrite-subterm term transform index)
  (if
    (= index 0)
    (or (transform term) term)
    (replace-subterms 
      term
      (car (foldl
             ; Finds the sub-term whose sub-tree contains the index we are
             ; looking for. The accumulator `result` is a pair
             ; (subterms . idx), where subterms is the list produced
             ; so far with updated subterms (one of them will have had
             ; a piece rewritten, by the end), and idx is the index
             ; of the subterm we're looking for, disregarding what we already
             ; looked at. idx will be set to #f when the term we wanted to
             ; rewrite was already rewritten.
             (lambda (st result)
               (let ([(sts . idx) result])
                 (if (not idx)
                   (cons (append sts (list st)) #f)
                   ; Calling term-size here makes the time complexity
                   ; quadratic. But since terms are tiny, this might not be
                   ; problematic. Linear-time solution would be to have
                   ; rewrite-subterm recursively compute the term size along
                   ; with the rewrite. Memoizing term-size also solves it.
                   (let ([s (term-size st)])
                     (if (< idx s)
                       ; The term to rewrite is in st - rewrite.
                       (cons (append sts 
                                     (list (rewrite-subterm st transform idx)))
                                     #f)
                       ; The term to rewrite is not in st - update index.
                       (cons (append sts (list st)) (- idx s)))))))
             (cons (list) (- index 1))
             (subterms term))))))

; Tells whether a binary operator is commutative: a op b = b op a
(define (is-commutative? op) (if (member op (list op+ op*)) #t #f))
; Tells whether operator op1 is associative with over op2: a op1 (b op2 c) = (a op1 b) op2 c
(define (is-associative? op1 op2)
  (or
    (and (eq? op1 op+) (or (eq? op2 op+) (eq? op2 op-)))
    (and (eq? op1 op*) (or (eq? op2 op*) (eq? op2 op/)))))
; Tells whether operator op1 distributes over op2: a op1 (b op2 c) = (a op1 b) op2 (a op1 c)
(define (is-distributive? op1 op2) 
  (and (or (eq? op1 op*) (eq? op1 op/))
       (or (eq? op2 op+) (eq? op2 op-))))

; Locally simplify the term with simple rewrite rules.
; These rules don't need to cover symmetric cases because we combine 
; them with random search to get a global simplification procedure.
; For example, they simplify (2*3)*x to 6x, but they don't match 2*(3x).
; However, during random search, we'll use multiplication's associativity
; to turn 2*(3*x) into (2*3)*x, and then the rule applies.
(define simpl-term-step
  (function
   ; Evaluate operation if both sides are numbers.
   [(BinOp op (Number x) (Number y))
    (Number (compute-bin-op op x y))]
   ; Add equal terms - case 1/3 (t + t --> 2x).
   [(BinOp op t t)
    #:if (eq? op op+)
    (BinOp op* (Number 2) t)]
   ; Add equal terms - case 2/3 (t + k*t --> (k+1)*t).
   [(BinOp op t (BinOp opx (Number k) t))
    #:if (and (eq? op op+) (eq? opx op*))
    (BinOp op* (Number (+ k 1)) t)]
   ; Add equal terms - case 3/3 (a*t + b*t --> (a + b)*t).
   [(BinOp op (BinOp opl a t) (BinOp opr b t))
    #:if (and (eq? op op+) (eq? opl op*) (eq? opr op*))
    (BinOp op* (BinOp op+ a b) t)]
   ; Add zero (t + 0 --> t).
   [(BinOp op t (Number n))
    #:if (and (eq? op op+) (eq? n 0))
    t]
   ; Multiply by one (t * 1 --> t).
   [(BinOp op t (Number n))
    #:if (and (eq? op op*) (eq? n 1))
    t]
   ; Multiply by zero (t * 0 --> 0).
   [(BinOp op t (Number n))
    #:if (and (eq? op op*) (eq? n 0))
    (Number 0)]
   ; Term minus itself.
   [(BinOp op t t)
    #:if (and (eq? op op-))
    (Number 0)]
   ; Recursively simplify.
   [(BinOp op t1 t2) (BinOp op (simpl-term-step t1) (simpl-term-step t2))]
   ; Default: do nothing.
   [t t]))

; Locally simplify term until a fixpoint is reached.
(define (simpl-term-local term)
  (let ([sterm (simpl-term-step term)])
    (if (equal? term sterm)
        term
        (simpl-term-local sterm))))

; Optimize a term using a black-box objective function using random search with a budget.
; Tries at most b random perturbations of the term (e.g. swapping addition order).
; (objective t) must give a number that will be *minimized* (e.g. term size).
(define (optimize-term term objective)
  (letrec
    ; p is the inverse of the probability of taking a random decision (see uses).
    ; This wasn't tuned, just works for simple examples.
    ([p 3]
     ; (random-neighbor p t) finds a random equivalent neighbor of the term t
     ; using probability 1/p to decide whether to apply a local rewrite.
     ; Thus, small p means more randomness.
     [random-neighbor
       (function
         ; Commutative operator.
         [(BinOp op l r)
          #:if (and (is-commutative? op) (= 0 (random p)))
          (BinOp op (random-neighbor r) (random-neighbor l))]
         ; Associative operator:  ((a op b) op c) --> (a op (b op c)).
         [(BinOp op (BinOp op2 a b) c)
          #:if (and (is-associative? op) (eq? op op2) (= 0 (random p)))
          (BinOp
            op
            (random-neighbor a)
            (BinOp op (random-neighbor b) (random-neighbor c)))]
         ; Associative operator:  (a op (b op c)) --> ((a op b) op c).
         [(BinOp op a (BinOp op2 b c))
          #:if (and (is-associative? op) (eq? op op2) (= 0 (random p)))
          (BinOp
            op
            (BinOp op (random-neighbor a) (random-neighbor b))
            (random-neighbor c))]
         ; Generic rule for binary operators.
         [(BinOp op l r)
          (BinOp op (random-neighbor l) (random-neighbor r))]
         ; TODO: apply distributive laws.
         ; Base case: don't do any transformation.
         [t t])]
     ; (random-step t p) finds a random neighbor of t. and returns a pair
     ; (t' . b), where b is the best of the two terms according to the
     ; objective function, and t' might be either t or t'.
     [random-step (lambda (t)
                    (let*
                      ([tr (simpl-term-local (random-neighbor t))]
                       [t-cost (objective t)]
                       [tr-cost (objective tr)]
                       [tr-better? (< tr-cost t-cost)])
                      ; If tr is smaller, always take it. Otherwise, take with
                      ; probability 1/p.
                      (if (or tr-better? (= 0 (random p)))
                        (cons tr (if tr-better? tr t))
                        (cons t t))))]
     [random-search-optimize (lambda (t best budget max-budget)
                               (if (= budget 0)
                                 best ; No more budget - give up.
                                 ; Otherwise, run a step and continue.
                                 (let* ([step-result (random-step t)]
                                        [tstep (car step-result)]
                                        [tstep-best (cdr step-result)]
                                        [next-best (if (< (objective tstep-best) 
                                                          (objective best)) 
                                                     tstep-best best)]
                                        [progress (not (eq? next-best best))])
                                   ; If made progress, reset budget, else decrement it.
                                   (random-search-optimize
                                     tstep
                                     next-best
                                     (if progress max-budget (- budget 1))
                                     max-budget))))]
     )
    (random-search-optimize term term 100 100)))

; Format an operator.
(define (format-op op)
  (match op
    [(== op+) "+"]
    [(== op-) "-"]
    [(== op*) "*"]
    [(== op/) "/"]
    ))

; Compact form of printing a term.
(define format-term
  (function
   ; AnyNumber
   [(AnyNumber) "?"]
   ; Number
   [(Number n) (format "~a" n)]
   ; Variable
   [(Variable v) (format "~a" v)]
   ; Variable with coefficient
   [(BinOp op (Number n) (Variable v)) #:if (eq? op op*) (format "~a~a" n v)]
   ; Generic binary operation.
   [(BinOp op a b) (format "(~a ~a ~a)" (format-term a) (format-op op) (format-term b))]
   ; Equality.
   [(Predicate 'Eq (a b))
    (format "~a = ~a" (format-term a) (format-term b))]
   ))

; Returns (format-term t) plus the raw representation of the term.
(define (format-term-debug t)
  (format "~a [~a]" (format-term t) t))

; Simplify a term: optimize the number of characters we need to write it down.
; This implicitly has many nice properties. Example: it prefers 2x instead of x*2,
; because 2x is formatted more compactly by format-term.
(define (simpl-term t)
  (optimize-term t (lambda (term) (string-length (format-term term)))))

; Show an example of simplification
(define (simpl-example t)
  (printf "~a simplifies to ~a\n" (format-term t) (format-term (simpl-term t))))

(provide
  simpl-term
  simpl-example
  format-term format-term-debug
  rewrite-subterm
  filter-subterms
  enumerate-subterms
  term-size
  goal-matches?
  Number Variable UnOp BinOp AnyNumber Predicate
  op+ op* op- op/ is-commutative? is-associative? is-distributive? compute-bin-op)
