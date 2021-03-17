const extractTernaryDigits = (digitsString) => {
  const digitsIt = digitsString.matchAll('[a-z][0-9]');
  const digits = [];

  while (true) {
    const next = digitsIt.next();
    if (next.done) {
      break;
    }
    const digit = next.value[0];
    digits.push(digit);
  }

  return digits;
}

const ternaryDigitsEqual = (s1, s2) => (extractTernaryDigits(s1).join("") ===
                                        extractTernaryDigits(s2).join(""));

export {
  extractTernaryDigits,
  ternaryDigitsEqual,
};
