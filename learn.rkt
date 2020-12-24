; Produce examples for learning a model that can solve a particular domain.

#lang racket

(require "solver.rkt")
(require "tactics.rkt")
(require "generation.rkt")
(require "serialize.rkt")
(require "facts.rkt")
(require "terms.rkt")
(require "value-function.rkt")

; Returns all facts in the SolverResult that are not part of the given solution.
(define (filter-irrelevant-facts sr solution)
  (filter (lambda (f) (not (findf (lambda (sf) (fact-terms-equal? sf f)) solution)))
          (SolverResult-facts sr)))

; Returns a list of negative examples of step-by-step solutions.
(define (make-negative-examples sr solution n)
  ; First, find facts that are not part of the given solution.
  (let* ([all-irrelevant-facts (filter-irrelevant-facts sr solution)]
         ; Then, take a random subset of n such facts.
         [picked-examples (take (shuffle all-irrelevant-facts)
                                (min n (length all-irrelevant-facts)))])
    ; Finally, take the step-by-step trace to each of the picked facts and
    ; format it.
    (map (lambda (f) (format-step-by-step
                      (trace-solver-steps (SolverResult-facts sr) (list f))
                      axiom->string))
         picked-examples)))

; Runs the solver until it is able to successfully solve n problems.
(define (run-solver-round
         generate-problem-fn
         strategy
         n-problems
         n-negative-examples
         ranking-fn
         beam-size
         depth
         solutions)
  (with-handlers ([exn? (lambda (e)
                          (printf "Error: ~a\n" e)
                          (run-solver-round generate-problem-fn
                                            strategy
                                            n-problems
                                            n-negative-examples
                                            ranking-fn
                                            beam-size
                                            depth
                                            solutions))])
  (if (= n-problems 0)
      solutions
      (let* ([problem (generate-problem-fn)]
             [_ (printf "Going to solve a problem...\n")]
             [sr (solve-problem problem
                                strategy
                                (lambda (all-facts facts)
                                  (take (ranking-fn all-facts facts) (min beam-size (length facts))))
                                depth)]
             [problem-solved? (empty? (SolverResult-unmet-goals sr))]
             [n-remaining (if problem-solved? (- n-problems 1) n-problems)]
             [_ (printf "~a, ~a problems left\n"
                        (if problem-solved? "suceeded" "timed out")
                        n-remaining)]
             [generated-example
              (if problem-solved?
                  (let* ([solution (get-step-by-step-solution sr)]
                         [negative-examples (make-negative-examples sr solution n-negative-examples)])
                    (list (hash 'type "Example"
                                'problem problem
                                'solution (format-step-by-step solution axiom->string)
                                'negative-examples negative-examples)))
                  (list))])

        (run-solver-round generate-problem-fn
                          strategy
                          n-remaining
                          n-negative-examples
                          ranking-fn
                          beam-size
                          depth
                          (append generated-example solutions))))))

(define (run-equation-solver-round
         n-problems
         depth
         n-round
         value-function)
  (let ([result (run-solver-round
                 generate-problem
                 s:equations
                 n-problems
                 5
                 (if value-function
                     rank-facts-value-function
                     (lambda (all-facts facts) (shuffle facts)))
                 30
                 depth
                 (list))]
        [output-file (format "equation-examples-round-~a.json" n-round)])
    (call-with-output-file output-file (lambda (out) (to-json result out)) #:exists 'replace)
    (printf "Wrote ~a\n" output-file)))

(define round-number (make-parameter 1))
(define n-problems (make-parameter 100))
(define depth (make-parameter 5))
(define use-value-function (make-parameter #f))

(command-line
 #:program "domain-learner"
 #:once-each
 [("-V" "--value-function") "Use learned value function server."
                            (use-value-function #t)]
 [("-n" "--round") n
                   "Round number <n> (determines output file)"
                   (round-number n)])

(run-equation-solver-round (n-problems) (depth) (round-number) (use-value-function))
