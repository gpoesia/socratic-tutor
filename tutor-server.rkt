#lang racket

(require net/url
         web-server/dispatch
         (prefix-in files: web-server/dispatchers/dispatch-files)
         (prefix-in filter: web-server/dispatchers/dispatch-filter)
         (prefix-in sequencer: web-server/dispatchers/dispatch-sequencer)
         web-server/servlet
         web-server/servlet-env
         web-server/dispatchers/filesystem-map
         web-server/http
         web-server/servlet-dispatch
         web-server/web-server)
(require json)
(require "terms.rkt")
(require "term-parser.rkt")
(require "serialize.rkt")
(require "solver.rkt")
(require "tactics.rkt")
(require "facts.rkt")
(require "questions.rkt")
(require "tutor.rkt")

(define (make-endpoint f)
  (lambda (req)
    (define binds (request-bindings/raw req))
    (define raw-params (bindings-assq #"params" binds))
    (let ([output
            (match raw-params
                   [(binding:form _ params-string)
                    (f (bytes->jsexpr params-string))]
                   [_ "error"])])
      (response/output
        (lambda (out)
          (displayln (to-json-string output) out))))))

(define api:parse-term
  (make-endpoint 
    (lambda (params)
      (parse-term (hash-ref params 'term)))))

(define api:leading-question
  (make-endpoint
    (lambda (params)
      (let* ([facts (map (compose assumption parse-term)
                         (hash-ref params 'facts))]
             [goals (map parse-term (hash-ref params 'goals))]
             [sr (find-solution goals facts s:equations
                                (prune:keep-smallest-k 20) 100)]
             [solved? (empty? (SolverResult-unmet-goals sr))]
             [step-by-step (and solved? 
                                (get-step-by-step-solution sr))])
        (printf "Goals: ~a\nFacts: ~a\n" 
                (map format-term goals)
                (map format-fact-v facts))
        (printf "Solved? ~a\n" solved?)
        (printf "Step by step: ~a\n" (length step-by-step))
        (if solved?
          (generate-leading-question
            (Fact-proof (first-non-assumption step-by-step))
            step-by-step)
          "Oh no, even I can't solve this!")))))

(define api:check
  (make-endpoint
    (lambda (params)
      (let* ([facts (map (compose assumption parse-term)
                         (hash-ref params 'facts))]
             [goals (map parse-term (hash-ref params 'goals))]
             [sr (find-solution goals facts s:all
                                (prune:keep-smallest-k 50) 10)]
             [contradiction (SolverResult-contradiction sr)])
        (if (not contradiction)
          (if (first-non-assumption (get-step-by-step-solution sr))
            "correct"
            "correct-finished")
          (let ([step-by-step (get-step-by-step-contradiction sr)])
            (generate-leading-question
              (Fact-proof (first-non-assumption step-by-step)) step-by-step)))))))

(define-values (routes reverse-uri)
  (dispatch-rules
    [("parse-term") api:parse-term]
    [("leading-question") api:leading-question]
    [("check") api:check]
    ))

(define (not-found req)
  (response/output
    (lambda (out)
      (displayln "Not found" out))))

(define stop
  (begin
    (printf "Running Web server on port 8123\n")
    (serve
      #:dispatch (sequencer:make
                   (dispatch/servlet routes)
                   (dispatch/servlet not-found))
      #:listen-ip #f
      #:port 8123)))

(with-handlers ([exn:break? (lambda (e)
                              (stop))])
               (sync/enable-break never-evt))
