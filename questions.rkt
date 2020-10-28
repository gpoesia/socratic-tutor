; Generate leading questions based on a solution step.

#lang racket

(require srfi/13)
(require "./terms.rkt")
(require "./facts.rkt")
(require "./tactics.rkt")

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

(provide generate-leading-question)
