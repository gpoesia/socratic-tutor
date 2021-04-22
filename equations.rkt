; Axioms and tactics used by the solver.
#lang algebraic/racket/base

(require algebraic/data)
(require algebraic/function)
(require algebraic/racket/base/forms)
(require racket/list)
(require racket/string)
(require racket/function)
(require racket/match)
(require racket/port)
(require rebellion/type/enum)
(require "terms.rkt")
(require "solver.rkt")
(require "term-parser.rkt")
(require "facts.rkt")
(require "debug.rkt")

; ==============================
; ========   Axioms  ===========
; ==============================
; Axioms are the building blocks of the solutions.

; Trivial axiom from which premises of the problem can be derived.
(define a:premise 'Premise)

; Flip an equality.
(define a:flip-equality
  (function
    [(Predicate 'Eq `(,t1 ,t2)) (Predicate 'Eq (list t2 t1))]
    [t #f]))

; Use equality e := t1 = t2 to substitute occurrences
; of the term t1 by the term t2 in equality e2.
(define (a:substitute-both-sides e)
  ((function
     [(Predicate 'Eq `(,t1 ,t2))
      (function
        [(Predicate 'Eq `(,t3 ,t4))
         (Predicate 'Eq (list (substitute-term t3 t1 t2)
                              (substitute-term t4 t1 t2)))]
        [t #f])
      ]
      [t (lambda (_) #f)])
   e))

; Commutes both sides of a binary operation.
(define a:commutativity?
  (function
    [(BinOp op _ _) (is-commutative? op)]
    [_ #f]))

(define a:commutativity
  (phi (BinOp op l r) (BinOp op r l)))

; Commutes two subtractions in a row: a - b - c => a - c - b
(define a:subtraction-commutativity?
  (function
    [(BinOp op (BinOp op a b) c) (eq? op op-)]
    [_ #f]))

(define a:subtraction-commutativity
  (phi (BinOp op (BinOp op a b) c) (BinOp op (BinOp op a c) b)))

; Computes a subtraction between equal terms, resulting in 0.
(define a:subtraction-same?
  (function
    [(BinOp (op #:if (eq? op op-)) a b) (equal? a b)]
    [_ #f]))

(define a:subtraction-same
  (phi (BinOp op a b) (Number 0)))

; Rearranges an associative operation.
(define a:associativity?
  (function
    [(BinOp op1 a (BinOp op2 b c)) (is-associative? op1 op2)]
    [(BinOp op1 (BinOp op2 a b) c) (is-associative? op2 op1)]
    [_ #f]))

(define a:associativity
  (function
    [(BinOp op1 a (BinOp op2 b c)) (BinOp op2 (BinOp op1 a b) c)]
    [(BinOp op1 (BinOp op2 a b) c) (BinOp op2 a (BinOp op1 b c))]
    [_ #f]))

; Evaluates a binary operation on numbers.
(define a:binop-eval?
  (function
    [(BinOp op (Number n1) (Number n2))
     #:if (not (and (eq? op op/) (= n2 0)))
     #t]
    [_ #f]))

(define a:binop-eval
  (phi (BinOp op (Number n1) (Number n2))
       (Number (compute-bin-op op n1 n2))))

; Simplifies adding/subtracting zero.
(define a:add-zero?
  (function
    [(BinOp (op #:if (eq? op op+)) (Number 0) t) #t]
    [(BinOp (op #:if (eq? op op+)) t (Number 0)) #t]
    [(BinOp (op #:if (eq? op op-)) t (Number 0)) #t]
    [t #f]))

(define a:add-zero
  (function
    [(BinOp (op #:if (eq? op op+)) (Number 0) t) t]
    [(BinOp (op #:if (eq? op op+)) t (Number 0)) t]
    [(BinOp (op #:if (eq? op op-)) t (Number 0)) t]
    [t #f]))

; Simplifies multiplication by zero.
(define a:mul-zero?
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 0) t) #t]
    [(BinOp (op #:if (eq? op op*)) t (Number 0)) #t]
    [t #f]))

(define a:mul-zero
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 0) t) (Number 0)]
    [(BinOp (op #:if (eq? op op*)) t (Number 0)) (Number 0)]
    [t #f]))

; Simplifies multiplication/division by one.
(define a:mul-one?
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 1) t) #t]
    [(BinOp (op #:if (eq? op op*)) t (Number 1)) #t]
    [(BinOp (op #:if (eq? op op/)) t (Number 1)) #t]
    [t #f]))

(define a:mul-one
  (function
    [(BinOp (op #:if (eq? op op*)) (Number 1) t) t]
    [(BinOp (op #:if (eq? op op*)) t (Number 1)) t]
    [(BinOp (op #:if (eq? op op/)) t (Number 1)) t]
    [t #f]))

; Applies distributivity law.
(define a:distributivity?
  (function
    [(BinOp op1 a (BinOp op2 b c))
     #:if (is-distributive? op1 op2)
     #t]
    [(BinOp op2 (BinOp op1 a1 b) (BinOp op1 a2 c))
     #:if (and (is-distributive? op1 op2) (equal? a1 a2))
     #t]
    [(BinOp op2 (BinOp op1 a b1) (BinOp op1 c b2))
     #:if (and (is-distributive? op1 op2) (equal? b1 b2))
     #t]
    [t #f]))

(define a:distributivity
  (function
    [(BinOp op1 a (BinOp op2 b c))
     #:if (is-distributive? op1 op2)
     (BinOp op2 (BinOp op1 a b) (BinOp op1 a c))]
    [(BinOp (BinOp op1 a1 b) (BinOp op2 a2 c))
     #:if (and (is-distributive? op1 op2) (equal? a1 a2))
     (BinOp op1 a1 (BinOp op2 b c))]
    [(BinOp op2 (BinOp op1 a b1) (BinOp op1 c b2))
     #:if (and (is-distributive? op1 op2) (equal? b1 b2))
     (BinOp op1 (BinOp op2 a c) b2)]
    [t #f]))

; Applies an operation op with term t to both sides of an equation.
(define a:op-both-sides
  (function*
    [((Predicate 'Eq (a b)) t op)
     (Predicate 'Eq (list (BinOp op a t) (BinOp op b t)))]
    [(_ _ _) #f]))

; ==============================
; ========   Tactics ===========
; ==============================
; Tactics apply axioms to the new known facts, producing others.

; When to apply t:flip
(define isolated-variable-t?
  (function
    [(Predicate 'Eq `(,t1 ,t2)) (or (Variable? t1) (Variable? t2))]
    [_ #f]))

(define (isolated-variable? f)
  (isolated-variable-t? (Fact-term f)))

; Apply a:flip-equality.
(define (t:flip unmet-goals old-facts new-facts)
  (map (lambda (f)
         (fact (a:flip-equality (Fact-term f))
               (FactProof a:flip-equality (list (FactId (Fact-id f))))))
       (filter isolated-variable? new-facts)))

; Meta-tactic that applies a simple term-level transform pair
; to all new facts, in all terms that satisfy the given predicate.
(define-syntax-rule (local-rewrite-tactic name predicate transform)
  (define (name unmet-goals old-facts new-facts)
    (apply append
      (map (lambda (f)
             (let ([indices (filter-subterms (Fact-term f) predicate)])
               (map (lambda (i)
                      (let ([rewritten (rewrite-subterm (Fact-term f) transform i)])
                        (log-debug "~a rewrote ~a => ~a\n"
                                   #(name)
                                   (format-term (Fact-term f))
                                   (format-term rewritten))
                        (if rewritten
                          (fact rewritten
                                (FactProof
                                  transform
                                  (list (FactId (Fact-id f)) i)))
                          #f
                          )))
                    indices)))
           new-facts))))

; Apply a:binop-eval.
(local-rewrite-tactic t:eval a:binop-eval? a:binop-eval)

; Apply a:associativity.
(local-rewrite-tactic t:associativity a:associativity? a:associativity)

; Apply a:commutativity.
(local-rewrite-tactic t:commutativity a:commutativity? a:commutativity)

; Apply a:subtraction-commutativity
(local-rewrite-tactic t:subtraction-commutativity
                      a:subtraction-commutativity?
                      a:subtraction-commutativity)

; Apply a:subtraction-same.
(local-rewrite-tactic t:subtraction-same a:subtraction-same? a:subtraction-same)

; Apply a:distributivity.
(local-rewrite-tactic t:distributivity a:distributivity? a:distributivity)

; Apply a:add-zero.
(local-rewrite-tactic t:add-zero a:add-zero? a:add-zero)

; Apply a:mul-zero.
(local-rewrite-tactic t:mul-zero a:mul-zero? a:mul-zero)

; Apply a:mul-one.
(local-rewrite-tactic t:mul-one a:mul-one? a:mul-one)

; Tactic that applies a:op-both-sides using terms from the equations.
(define (produce-new-equalities f)
  (match f
    [(Fact id t proof)
     ((function
       [(Predicate 'Eq (a b))
        (let ([all-terms (filter (lambda (t) (= 1 (term-size t)))
                                 (append (enumerate-subterms a) (enumerate-subterms b)))])
          (map
            (lambda (t-op)
              (let
                ([np (a:op-both-sides t (car t-op) (cadr t-op))])
                (log-debug "#(t:apply-op-both-sides) rewrote ~a => ~a\n"
                           (format-term t)
                           (format-term np))
                (fact np (FactProof a:op-both-sides
                                    (list (FactId id) (car t-op) (cadr t-op))))))
            (cartesian-product all-terms (list op- op/ op+))))
        ]
       [_ empty]
       ) t)]
    [_ empty]))

(define (t:apply-op-both-sides unmet-goals old-facts new-facts)
  (apply append (map produce-new-equalities new-facts)))

; Tactic that substitutes one equation into another.
(define (t:substitute unmet-goals old-facts new-facts)
  (apply append
    (map (lambda (new-fact)
           (let ([sub (a:substitute-both-sides (Fact-term new-fact))])
             (map (lambda (other-fact)
                    (fact (sub (Fact-term other-fact))
                          (FactProof a:substitute-both-sides
                                     (list (FactId (Fact-id new-fact))
                                           (FactId (Fact-id other-fact))))))
                  (append new-facts old-facts))))
         (filter isolated-variable? new-facts))))

(define (combine-tactics tactics)
  (lambda (unmet-goals old-facts new-facts)
      (apply append
             (map (lambda (t) (t unmet-goals old-facts new-facts))
                  tactics))))

; Applies all tactics.
(define t:all (combine-tactics
                (list
                  t:flip
                  t:substitute
                  t:eval
                  t:associativity
                  t:commutativity
                  t:subtraction-commutativity
                  t:subtraction-same
                  t:distributivity
                  t:add-zero
                  t:mul-zero
                  t:mul-one
                  t:apply-op-both-sides
                  )))

; ==============================
; ======== Domain =============
; ==============================

(define MAX-SIZE 30)

; Function that, given a node, lists all child nodes.
(define (d:equations facts)
  ; Avoid huge equations.
  (if (> (term-size (Fact-term (last facts))) MAX-SIZE)
      empty
      (filter (lambda (f) (not (member f facts fact-terms-equal?)))
              (t:all #f empty (list (last facts))))))

; ==============================
; ======== Strategies ==========
; ==============================
; These guide the solver in applying tactics.

; Trivial strategy: always apply all known tactics.
(define (s:all old-facts last-facts unmet-goals strategy-state)
  (values (t:all unmet-goals old-facts last-facts)
          (append old-facts last-facts)
          #f))

; Applies tactics to try to solve linear equations using a common strategy.
; Algorithm:
; 1- Apply t:eval, t:associativity, t:commutativity, t:subtraction-commutativity,
;          t:distributivity, t:add-zero, t:mul-zero, t:mul-one, until they don't
;          produce any more facts.
; 2- Get only the top k simplest facts found
; 3- Apply one round of apply-op-both-sides
; 4- Repeat (1-3) until it managed to isolate a variable. Either we reached
;    (a) v = number, or (b) v = <some function of other variables>.
;    If (a), we're done; otherwise, apply one round of t:substitute and
;    repeat from (1).

(define-enum-type s:equations-state (st:simpl st:filter st:op-both-sides st:substitute))

(define t:equations-simpl
  (combine-tactics
    (list
      t:eval
      t:associativity
      t:commutativity
      t:subtraction-commutativity
      t:subtraction-same
      t:distributivity
      t:flip
      t:add-zero
      t:mul-zero
      t:mul-one)))

(define (smallest-k-facts k facts)
  (take (sort-facts-by-size facts) (min k (length facts))))

; Look for facts that meet isolated-variable? and that have not been used yet.
(define (any-new-isolated-variable? facts)
  (ormap
    (lambda (f)
      (and
        (isolated-variable? f)
        (let ([id (Fact-id f)])
              (andmap (lambda (of) (not (member id (fact-dependencies of))))
                      facts))))
    facts))

(define is-trivial?
  (function
    [(Predicate 'Eq `(,a ,b)) (equal? a b)]
    [_ #f]))

(define (remove-trivial facts)
  (filter (lambda (f) (not (is-trivial? (Fact-term f)))) facts))

(define (s:equations old-facts last-facts unmet-goals last-state)
  (let* ([state (cond
                  [(and (equal? last-state st:simpl)
                        (> 0 (length last-facts))) st:simpl]
                  [(and (equal? last-state st:simpl)
                        (equal? 0 (length last-facts))) st:filter]
                  [(and (equal? last-state st:filter)
                        (any-new-isolated-variable? last-facts)) st:substitute]
                  [(equal? last-state st:filter) st:op-both-sides]
                  [(equal? last-state st:substitute) st:op-both-sides]
                  [(equal? last-state st:op-both-sides) st:simpl]
                  [else st:simpl])]
         [new-facts
           (match state
             [(== st:simpl) (t:equations-simpl unmet-goals old-facts last-facts)]
             [(== st:filter) (smallest-k-facts 50 (append old-facts last-facts))]
             [(== st:op-both-sides) (t:apply-op-both-sides unmet-goals old-facts last-facts)]
             [(== st:substitute) (t:substitute unmet-goals old-facts last-facts)])]
         [kept-existing-facts
           (match state
             [(or (== st:simpl) (== st:op-both-sides) (== st:substitute))
              (append old-facts last-facts)]
             [(== st:filter) empty])]
         )
    (values (remove-trivial new-facts) kept-existing-facts state)))

; Load templates at start-up.
(define cognitive-tutor-templates
  (let*
      ([f (open-input-file "./data/cognitive_tutor_templates.txt")]
       [c (port->string f)]
       [templates-str (string-split c "\n")]
       [templates (map parse-term templates-str)])
    templates))

; Returns a generator that draws a random template from ts and randomizes its constants.
(define (generator-from-templates ts)
  (lambda ()
    (let* ([equation (randomize-constants (list-ref ts (random 0 (length ts))))]
           [variables (remove-duplicates
                       (remove* '(#f)
                                (map (function
                                      [(Variable v) v]
                                      [_ #f])
                                     (enumerate-subterms equation))))]
           [goals (map (lambda (v) (Predicate 'Eq (list (Variable v)
                                                        (AnyNumber))))
                       variables)])
      (Problem
       (list (assumption equation))
       goals))))

; Substitutes constants by random positive numbers, and keeps everything else intact.
(define (randomize-constants t)
  (map-subterms
   (function
    [(Number n) (Number (random 1 11))]
    [t t])
   t))

(provide
  a:premise
  a:flip-equality
  a:commutativity
  a:subtraction-commutativity
  a:subtraction-same
  a:binop-eval
  a:associativity
  a:add-zero
  a:mul-zero
  a:mul-one
  a:distributivity
  a:op-both-sides
  a:substitute-both-sides

  s:all
  s:equations

  d:equations
  generator-from-templates
  cognitive-tutor-templates
  )
