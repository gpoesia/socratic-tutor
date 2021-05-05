import _ from 'lodash';

function parseTerm(s, i) {
  if (s[i] == " ") {
    return parseTerm(s, i+1);
  }

  if (s[i] == '-' || !isNaN(s[i])) {
    let token = "";
    if (s[i] == s[i
}

export function formatTerm(t, inBinaryOp=false) {
  if (!isNaN(t)) {
    const n = _.toNumber(t);
    if (n < 0 && inBinaryOp) {
      return '(' + n + ')';
    }
    return n;
  } else {
    let op = null;
    let depth = 0;

    for (let i = 0; i < t.length; i++) {
      if (t[i] == '(') {
        depth += 1;
      } else if (t[i] == ')') {
        depth -= 1;
      } else if ('+-*/='.indexOf(t[i]) != -1) {
        opIndex = i;
      }
    }
  }
}
