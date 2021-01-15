#lang racket

(require racket/date)

(require gregor)
(require gregor/period)

(define (pad-number n w)
  (~a n
      #:min-width w
      #:align 'right
      #:pad-string "0"))

(define (format-period period)
  (let* ([h (period-ref period 'hours)]
         [m (period-ref period 'minutes)]
         [s (period-ref period 'seconds)])
    (format "~a:~a:~a" (pad-number h 2) (pad-number m 2) (pad-number s 2))))

(define (repeat-string s n)
  (string-join (map (lambda (_) s) (range n)) ""))

; Returns a progress bar for a number p between 0.0 and 1.0.
(define (progress-bar p begin-time n-steps [width 50])
  (let* ([n-filled (exact-round (* p width))]
         [n-spaces (- width n-filled)]
         [time-delta (- (current-seconds) begin-time)]
         [total-time (/ time-delta (max 0.0001 p))]
         [eta-period (/ time-delta (max 0.0001 p))])
    (format "[~a~a] ~a/s, ETA: ~a"
            (repeat-string "=" n-filled)
            (repeat-string " " n-spaces)
            (real->decimal-string (/ n-steps time-delta))
            (format-period (time-period-between (posix->datetime begin-time)
                                                (posix->datetime (+ begin-time total-time)))))))

(provide
  progress-bar
  repeat-string
  format-period)
