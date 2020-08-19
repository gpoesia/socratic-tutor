#lang racket

(require "./terms.rkt")

(simpl-example (BinOp op+ (Number 2) (Number 4)))
(simpl-example (BinOp op+ (Number 2) (Variable 'x)))
(simpl-example (BinOp op+ (Variable 'x) (Variable 'x)))
(simpl-example (BinOp op+
                      (BinOp op+ (Variable 'x) (Variable 'x))
                      (BinOp op+ (Variable 'x) (BinOp op+ (Variable 'y) (Variable 'x)))))
(simpl-example (BinOp op+ (BinOp op+ (Number 2) (Variable 'x)) (Variable 'x)))
(simpl-example (BinOp op* (Number 3) (BinOp op+ (Number 2) (Number 4))))
(simpl-example (BinOp op+
                      (BinOp op+ (Number 4) (Variable 'x))
                      (BinOp op+ (Variable 'x) (Number 4))))

(simpl-example (BinOp op-
                 (BinOp op- (BinOp op+ (BinOp op* (Number 2) (Variable 'x)) (Number 4)) (Variable 'x))
                     (Variable 'x)))

(simpl-example (BinOp op+
                 (BinOp op+ (BinOp op+ (BinOp op* (Number 2) (Variable 'x)) (Number 4))
                        (BinOp op* (Number -1) (Variable 'x)))
                 (BinOp op* (Number -1) (Variable 'x))))
