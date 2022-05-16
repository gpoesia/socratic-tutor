## Debug! Successes are fake...

Example "solution" to a problem:
```
[[State(((5 + -3x) - (-5)) = -1x),
State((((5 + -3x) - (-5)) / x) = ((-1) * (x / x)) | div-assoc x; 10, (-1x / x)),
State(((((5 + -3x) - (-5)) / x) / (-3)) = ((-1) * ((x / x) / (-3))) | div-assoc (-3); 12, (((-1) * (x / x)) / (-3))),
State((((5 + (-3x - (-5))) / x) / (-3)) = (((x / x) / (-3)) * (-1)) | comm-assoc 12, ((-1) * ((x / x) / (-3))); 3, ((5 + -3x) - (-5))),
State(((((5 + (-3x - (-5))) / x) / (-3)) + 5) = ((((x / x) / (-3)) * (-1)) + 5) | add 5),
State((((((5 + (-3x - (-5))) / x) / (-3)) + 5) / (-1)) = ((5 + (((x / x) / (-3)) * (-1))) / (-1)) | comm-div 14, ((((x / x) / (-3)) * (-1)) + 5); (-1))]]
```
Replicated by running `python environment.py --interact --rust` on the problem `((5 + -3x) - (-5)) = -1x` with the following choices of next actions: 21, 5, 20, 5, 3, 4, 8, 4, 33. After that, the environment gives no choices for next action.

Thus, we have to fix stepping so that it doesn't return `None` in this case.