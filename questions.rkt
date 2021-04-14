; Generate leading questions based on a solution step.

#lang racket

(require srfi/13)
(require "./terms.rkt")
(require "./facts.rkt")
(require "./equations.rkt")
(require "./ternary.rkt")
(require "./sorting.rkt")

; Simple utilities to make it easier to access proof parameters.
(define (first-param proof)
  (first (FactProof-parameters proof)))

(define (second-param proof)
  (second (FactProof-parameters proof)))

(define (third-param proof)
  (third (FactProof-parameters proof)))

(define (find-fact f-id facts)
  (findf (lambda (f) (equal? (Fact-id f) (FactId-id f-id))) facts))

(define (generate-leading-question fact-proof all-facts)
  (define-syntax-rule (local-rewrite-question q)
    (let* ([f (find-fact (first-param fact-proof) all-facts)]
           [t (generate-term-boundary-string
                f
                (second-param fact-proof))])
      (format q (Fact-id f) (format-fact-i f) t)))

  (match (FactProof-axiom fact-proof)
    [(or (== a:commutativity) (== a:subtraction-commutativity))
     (local-rewrite-question
       "What do you get if you take (~a) and swap both sides of the operation below?\n~a\n~a\n")]
    [(== a:binop-eval)
     (local-rewrite-question
       "What do you get if you take (~a) and calculate the operation below?\n~a\n~a\n")]
    [(== a:associativity)
     (local-rewrite-question
       "What do you get if you take (~a) and rearrange the parenthesis in the operation below?\n~a\n~a\n")]
    [(== a:add-zero)
     (local-rewrite-question
       "What do you get if you take (~a) and get rid of the addition by zero in the operation below?\n~a\n~a\n")]
    [(== a:mul-zero)
     (local-rewrite-question
       "What do you get if you take (~a) and calculate the multiplication by zero in the operation below?\n~a\n~a\n")]
    [(== a:mul-one)
     (local-rewrite-question
       "What do you get if you take (~a) and get rid of the multiplication by one in the operation below?\n~a\n~a\n")]
    [(== a:distributivity)
     (local-rewrite-question
       "What do you get if you take (~a) and rearrange the operation below using distributivity?\n~a\n~a\n")]
    [(== a:op-both-sides)
     (format
       "What do you get if you take:\n~a\nand ~a ~a on both sides?\n"
       (format-fact-i (find-fact (first-param fact-proof) all-facts))
       (operation-name (third-param fact-proof))
       (format-term (second-param fact-proof)))]
    [(== a:substitute-both-sides)
     (let ([source (find-fact (first-param fact-proof) all-facts)]
           [target (find-fact (second-param fact-proof) all-facts)])
       (format
         "What do you get if you use\n~a\n to substitute ~a by ~a in ~a?\n"
         (format-fact-i source)
         (format-term (first (Predicate-terms (Fact-term source))))
         (format-term (second (Predicate-terms (Fact-term source))))
         (format-fact-i target)))]
    [_ #f]))

(define (generate-step-description fact-proof all-facts)
  (define-syntax-rule (local-rewrite-description d)
    (let* ([f (find-fact (first-param fact-proof) all-facts)]
           [t (get-term-by-index (Fact-term f) (second-param fact-proof))])
      (format d (format-term t))))

  (match (FactProof-axiom fact-proof)
    [(== 'Assumption)
     "Given"]
    [(or (== a:commutativity) (== a:subtraction-commutativity))
     (local-rewrite-description "Change the order of operations in ~a")]
    [(== a:binop-eval)
     (local-rewrite-description "Calculate ~a")]
    [(== a:associativity)
     (local-rewrite-description "Rearrange the parentheses in ~a")]
    [(== a:add-zero)
     "Use that adding zero doesn't change the result"]
    [(== a:mul-zero)
      "Use that multiplying by zero gives zero"]
    [(== a:mul-one)
      "Use that multiplying by one doesn't change the result"]
    [(== a:subtraction-same)
      "Use that anything minus itself is zero."]
    [(== a:flip-equality)
      "Flip the sides of the equation"]
    [(== a:distributivity)
     (local-rewrite-description "Apply the distributivity law in ~a")]
    [(== a:op-both-sides)
     (format
       "~a ~a on both sides"
       (operation-name-upcase (third-param fact-proof))
       (format-term (second-param fact-proof)))]
    [(== a:substitute-both-sides)
     (let ([source (find-fact (first-param fact-proof) all-facts)])
       (format
         "Substitute ~a by ~a"
         (format-term (first (Predicate-terms (Fact-term source))))
         (format-term (second (Predicate-terms (Fact-term source))))))]
    [_ ""]))

(define (generate-formal-step-description fact-proof all-facts)
  ; Template for describing a term-local transformation in equations.
  (define-syntax-rule (local-rewrite-description d)
    (let* ([f (find-fact (first-param fact-proof) all-facts)]
           [index (second-param fact-proof)]
           [t (get-term-by-index (Fact-term f) index)])
      (format d index (format-term t))))

  ; Template for describing a transformation in ternary-addition.
  (define-syntax-rule (ternary-rewrite-description d use-second?)
    (let* ([f (find-fact (first-param fact-proof) all-facts)]
           [f-digits (TernaryNumber-digits (Fact-term f))]
           [index (second-param fact-proof)]
           [first-digit (format-term (list-ref f-digits index))]
           [second-digit (if (< (+ 1 index) (length f-digits))
                             (format-term (list-ref f-digits (+ 1 index)))
                             "")])
      (apply format (append (list d index first-digit)
                            (if use-second? (list second-digit) empty)))))

  (match (FactProof-axiom fact-proof)
    [(== 'Assumption)
     "given"]

    ; ================
    ; Equations domain
    ; ================
    [(or (== a:commutativity) (== a:subtraction-commutativity))
     (local-rewrite-description "comm ~a[~a]")]
    [(== a:binop-eval)
     (local-rewrite-description "eval ~a[~a]")]
    [(== a:associativity)
     (local-rewrite-description "assoc ~a[~a]")]
    [(== a:add-zero)
     (local-rewrite-description "add0 ~a[~a]")]
    [(== a:mul-zero)
     (local-rewrite-description "mul0 ~a[~a]")]
    [(== a:mul-one)
     (local-rewrite-description "mul1 ~a[~a]")]
    [(== a:distributivity)
     (local-rewrite-description "dist ~a[~a]")]
    [(== a:flip-equality)
     "symm"]
    [(== a:op-both-sides)
     (format
       "~a ~a"
       (operation-formal-name(third-param fact-proof))
       (format-term (second-param fact-proof)))]
    [(== a:substitute-both-sides)
     (let ([source (find-fact (first-param fact-proof) all-facts)])
       (format
         "sub ~a => ~a"
         (format-term (first (Predicate-terms (Fact-term source))))
         (format-term (second (Predicate-terms (Fact-term source))))))]

    ; ==============
    ; Ternary domain
    ; ==============
    [(== td:swap)
     (ternary-rewrite-description "swap ~a[~a ~a]" #t)]
    [(== td:add-consecutive)
     (ternary-rewrite-description "comb ~a[~a ~a]" #t)]
    [(== td:erase-zero)
     (ternary-rewrite-description "del ~a[~a]" #f)]

    ; ==============
    ; Sorting domain
    ; ==============
    [(== sd:reverse)
     "rev"]
    [(== sd:swap)
     (format "swap ~a" (list-ref (FactProof-parameters fact-proof) 1))]

    [_ ""]))

(define (generate-term-boundary-string fact t-idx [fmt format-fact-i])
  (let* ([marked-fact (Fact (Fact-id fact)
                            (mark-term (Fact-term fact) t-idx)
                            (Fact-proof fact))]
         [marked-str (fmt marked-fact)]
         [begin-pos (string-contains marked-str BEGIN-MARKER)]
         [end-pos (- (string-contains marked-str END-MARKER)
                     (string-length BEGIN-MARKER))])
    (string-append (make-string begin-pos #\ )
                   (make-string (- end-pos begin-pos) #\-))))

; Format an operator.
(define (operation-name op)
  (match op
    [(== op+) "add"]
    [(== op-) "subtract"]
    [(== op*) "multiply by"]
    [(== op/) "divide by"]
    ))

; Format an operator for a formal step description.
(define (operation-formal-name op)
  (match op
    [(== op+) "add"]
    [(== op-) "sub"]
    [(== op*) "mul"]
    [(== op/) "div"]
    ))

(define (operation-name-upcase op)
  (match op
    [(== op+) "Add"]
    [(== op-) "Subtract"]
    [(== op*) "Multiply by"]
    [(== op/) "Divide by"]
    ))

(provide
  generate-leading-question
  generate-step-description
  generate-formal-step-description)
