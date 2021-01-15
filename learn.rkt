; Produce examples for learning a model that can solve a particular domain.

#lang racket

(require racket/date)
(require racket/engine)
(require racket/place)

(require "solver.rkt")
(require "tactics.rkt")
(require "generation.rkt")
(require "serialize.rkt")
(require "facts.rkt")
(require "terms.rkt")
(require "value-function.rkt")
(require "debug.rkt")
(require "util.rkt")

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

(define SOLVER-TIMEOUT 60)

(define (generate-and-solve-problems
         channel
         generator-name
         strategy-name
         ranking-fn-name
         beam-size
         depth
         n-negative-examples)
 (let* ([generate-problem-fn (get-problem-generator-by-name generator-name)]
        [strategy (get-strategy-by-name strategy-name)]
        [ranking-fn (get-ranking-fn-by-name ranking-fn-name)]
        [problem (generate-problem-fn)]
        [e (engine (lambda (_) (solve-problem
                                problem
                                strategy
                                (lambda (all-facts facts)
                                  (take (ranking-fn all-facts facts) (min beam-size (length facts))))                 
                                depth)))]
        [success? (engine-run (* 1000 SOLVER-TIMEOUT) e)]
        [sr (if success? (engine-result e) #f)])
    (place-channel-put
      channel
      (to-jsexpr
       (if (and success? (problem-solved? sr))
        (let* ([solution (get-step-by-step-solution sr)]
               [negative-examples (make-negative-examples sr solution n-negative-examples)])
          (hash 'type "Example"
                'success #t
                'problem problem
                'solution (format-step-by-step solution axiom->string)
                'negative-examples negative-examples))
        (hash 'type "Example"
              'success #f
              'problem problem))))
    (generate-and-solve-problems channel generator-name strategy-name
                                 ranking-fn-name beam-size depth n-negative-examples)))

; Runs the solver until it is able to successfully solve n problems.
(define (run-solver-round
         n-threads
         begin-time
         total-problems
         solver-places
         place-dead-evts
         generate-problem-fn
         strategy-name
         n-problems
         n-negative-examples
         ranking-fn-name
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
                   generate-problem-fn
                   strategy-name
                   ranking-fn-name
                   beam-size
                   depth
                   n-negative-examples))])
          (log-debug "Starting new solver thread...\n")
          (run-solver-round
            n-threads
            begin-time
            total-problems
            (cons p solver-places)
            (cons (place-dead-evt p) place-dead-evts)
            generate-problem-fn
            strategy-name
            n-problems
            n-negative-examples
            ranking-fn-name
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
            (run-solver-round
              n-threads
              begin-time
              total-problems
              (remove (list-ref solver-places i) solver-places)
              (remove next-evt place-dead-evts)
              generate-problem-fn
              strategy-name
              n-problems
              n-negative-examples
              ranking-fn-name
              beam-size
              depth
              solutions))
        ; Otherwise, it's a new example. Append it and continue.
          (let* ([success? (hash-ref next-evt 'success)]
                 [remaining-problems (- n-problems (if success? 1 0))])
            (if success?
              (begin (printf "\r~a | ~a/~a solved, ~a attempts"
                             (progress-bar (/ (- total-problems remaining-problems) remaining-problems)
                                           begin-time)
                             (- total-problems remaining-problems)
                             total-problems
                             (+ 1 (length solutions)))
                     (flush-output))
              (void))
            (run-solver-round
             n-threads
             begin-time
             total-problems
             solver-places
             place-dead-evts
             generate-problem-fn
             strategy-name
             remaining-problems
             n-negative-examples
             ranking-fn-name
             beam-size
             depth
             (cons next-evt solutions)))))])))

(define (run-equation-solver-round
         n-problems
         depth
         negative-examples
         beam-width
         output-file
         value-function)
  (let ([result (run-solver-round
                 (min 16 (processor-count))
                 (current-seconds)
                 n-problems
                 (list)
                 (list)
                 'equations:gen
                 's:all
                 n-problems
                 negative-examples
                 (if value-function
                     'rank-facts-value-function
                     'smallest)
                 beam-width
                 depth
                 (list))])
    (call-with-output-file output-file (lambda (out) (to-json result out)) #:exists 'replace)
    (printf "\nWrote ~a\n" output-file)))

(define (get-ranking-fn-by-name name)
  (match name
    [(== 'smallest) (lambda (all-facts facts) (sort-facts-by-size facts))]
    [(== 'shuffle) (lambda (all-facts facts) (shuffle facts))]
    [(== 'rank-facts-value-function) rank-facts-value-function]))

(define (get-strategy-by-name name)
  (match name
    [(== 's:all) s:all]))

(define (get-problem-generator-by-name name)
  (match name
    [(== 'equations:gen) generate-problem]))

(provide
  run-equation-solver-round)
