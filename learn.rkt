; Produce examples for learning a model that can solve a particular domain.

#lang racket

(require racket/date)
(require racket/engine)
(require racket/place)

(require "solver.rkt")
(require "equations.rkt")
(require "generation.rkt")
(require "domains.rkt")
(require "serialize.rkt")
(require "facts.rkt")
(require "terms.rkt")
(require "questions.rkt")
(require "value-function.rkt")
(require "debug.rkt")
(require "util.rkt")

; Returns all facts in the SolverResult that are not part of the given solution.
(define (filter-irrelevant-facts sr solution)
  (filter (lambda (f) (not (findf (lambda (sf) (fact-terms-equal? sf f)) solution)))
          (SolverResult-facts sr)))

; Returns a list of negative examples of step-by-step solutions.
(define (make-negative-examples solution all-nodes)
  ; For each step i in the solution (skipping the first),
  ; find a node that (1) has exactly i steps,
  ; (2) is equal to the solution up to step (i-1), and
  ; (3) differs in step i.
  (map (lambda (node)
         (hash 'index (- (length (MCTSNode-facts node)) 1)
               'step (format-term (Fact-term (last (MCTSNode-facts node))))
               'step-tex (format-term-tex (Fact-term (last (MCTSNode-facts node))))
               'step-description (generate-step-description
                                   (Fact-proof (last (MCTSNode-facts node)))
                                   (MCTSNode-facts node))
               'step-formal-description (generate-formal-step-description
                                         (Fact-proof (last (MCTSNode-facts node)))
                                         (MCTSNode-facts node))
               'value (MCTSNode-value node)))
    (filter identity
      (map (lambda (n-steps)
             (let ([candidates (filter (lambda (node)
                                         (and
                                           (= (length (MCTSNode-facts node)) n-steps) ; (1)
                                           (= (- n-steps 1)                           ; (2) and (3)
                                              (length (take-common-prefix
                                                        (MCTSNode-facts node)
                                                        solution
                                                        fact-terms-equal?)))))
                                       all-nodes)])
             (car (append (shuffle candidates) (list #f)))))
           (range 2 (+ 1 (length solution)))))))

; Returns a list of values of each step in the solution.
(define (get-solution-value solution all-nodes)
  ; For each step i in the solution,
  ; find a node that (1) has exactly i steps,
  ; (2) is equal to the solution up to step i.
  (map (lambda (n-steps)
    (MCTSNode-value (findf (lambda (node)
                            (= (length (MCTSNode-facts node)) n-steps) ; (1)
                            (= n-steps ; (2)
                                     (length (take-common-prefix
                                              (MCTSNode-facts node)
                                              solution
                                              fact-terms-equal?))))
                            all-nodes)))
    (range 1 (+ 1 (length solution)))))

(define SOLVER-TIMEOUT 120)

(define (generate-and-solve-problems
         channel
         domain-name
         policy-name
         policy-args
         beam-size
         depth
         n-negative-examples)
 (let* ([domain (get-domain-by-name domain-name)]
        [generate-problem-fn (Domain-generator domain)]
        [policy (get-policy-by-name policy-name policy-args)]
        [problem (generate-problem-fn)]
        [e (engine (lambda (_) (solve-problem-smc
                                problem
                                domain
                                policy
                                beam-size
                                depth)))]
        [success? (engine-run (* 1000 SOLVER-TIMEOUT) e)]
        [result (if success? (engine-result e) #f)])
    (place-channel-put
      channel
      (to-jsexpr
       (if (and success? (MCTSResult-terminal result))
        (let* ([solution (MCTSNode-facts (MCTSResult-terminal result))]
               [negative-examples (make-negative-examples solution (MCTSResult-nodes result))])
          (hash 'type "Example"
                'success #t
                'problem problem
                'solution-detailed (format-step-by-step solution axiom->string)
                'solution-formal-description (map
                                              (lambda (f)
                                                (generate-formal-step-description
                                                 (Fact-proof f) solution))
                                              solution)
                'solution-description (map
                                        (lambda (f)
                                          (generate-step-description
                                          (Fact-proof f) solution))
                                        solution)
                'solution (format-step-by-step-terms solution)
                'solution-tex (format-step-by-step-terms-tex solution)
                'solution-value (get-solution-value solution (MCTSResult-nodes result))
                'negative-examples negative-examples))
        (hash 'type "Example"
              'success #f
              'problem problem))))
    (generate-and-solve-problems channel domain-name
                                 policy-name policy-args beam-size depth n-negative-examples)))

; Runs the solver until it is able to successfully solve n problems.
(define (run-solver-loop
         n-threads
         begin-time
         total-problems
         solver-places
         place-dead-evts
         domain-name
         n-problems
         n-negative-examples
         policy-name
         policy-args
         beam-size
         depth
         solutions)
  (with-handlers ([exn:break? (lambda (e)
                                (begin (printf "Stopping...\n")
                                       solutions))]
                  [exn? (lambda (e) (begin (printf "Error: ~a\n" e)
                                           solutions))])
    (cond
      ; If we solved enough problems.
      [(<= n-problems 0) solutions]
      ; If we have budget to create more threads.
      [(< (length solver-places) n-threads)
       (let ([p (place/context ch
                  (generate-and-solve-problems
                   ch
                   domain-name
                   policy-name
                   policy-args
                   beam-size
                   depth
                   n-negative-examples))])
          (log-debug "Starting new solver thread...\n")
          (run-solver-loop
            n-threads
            begin-time
            total-problems
            (cons p solver-places)
            (cons (place-dead-evt p) place-dead-evts)
            domain-name
            n-problems
            n-negative-examples
            policy-name
            policy-args
            beam-size
            depth
            solutions))]
     ; Otherwise: wait on any of the places.
     [#t (let ([next-evt (sync (apply choice-evt (append solver-places place-dead-evts)))])
        ; If next-evt is evt?, one of the places is dead: find it and
        ; remove from the list.
        (if (evt? next-evt)
          (let ([i (index-of place-dead-evts next-evt)])
            (log-debug "Solver thread ~a died.\n" i)
            (run-solver-loop
              n-threads
              begin-time
              total-problems
              (remove (list-ref solver-places i) solver-places)
              (remove next-evt place-dead-evts)
              domain-name
              n-problems
              n-negative-examples
              policy-name
              policy-args
              beam-size
              depth
              solutions))
        ; Otherwise, it's a new example. Append it and continue.
          (let* ([success? (hash-ref next-evt 'success)]
                 [remaining-problems (- n-problems (if success? 1 0))])
            (if success?
              (begin (printf "\r~a | ~a/~a solved, ~a attempts"
                             (progress-bar (/ (- total-problems remaining-problems) total-problems)
                                           begin-time
                                           (- total-problems remaining-problems))
                             (- total-problems remaining-problems)
                             total-problems
                             (+ 1 (length solutions)))
                     (flush-output))
              (void))
            (run-solver-loop
             n-threads
             begin-time
             total-problems
             solver-places
             place-dead-evts
             domain-name
             remaining-problems
             n-negative-examples
             policy-name
             policy-args
             beam-size
             depth
             (cons next-evt solutions)))))])))

(define (run-solver-round
         domain-name
         n-problems
         depth
         negative-examples
         beam-width
         output-file
         policy-name
         policy-args
         max-threads)
  (let ([result (run-solver-loop
                 (min max-threads (processor-count))
                 (current-seconds)
                 n-problems
                 (list)
                 (list)
                 domain-name
                 n-problems
                 negative-examples
                 policy-name
                 policy-args
                 beam-width
                 depth
                 (list))])
    (call-with-output-file output-file (lambda (out) (to-json result out)) #:exists 'replace)
    (printf "\nWrote ~a\n" output-file)))

(define (get-policy-by-name name args)
  (match name
    ["smallest" inverse-term-size-value-function]
    ["random" random-value-function]
    ["neural" (apply make-neural-value-function args)]))

(provide
  run-solver-round)
