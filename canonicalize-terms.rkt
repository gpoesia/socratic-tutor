#lang racket

(require "terms.rkt")
(require "term-parser.rkt")

(define (map-lines ip f)
  (let ([l (read-line ip 'any)])
    (if (eof-object? l)
        empty
        (cons (f l) (map-lines ip f)))))

(define (main)
  (map-lines
    (open-input-file "input.txt")
    (lambda (l)
        (printf "~a\n" (format-term (parse-term l)))))
  (void))

(provide main)
