import React from "react";
import { colors } from 'just-mix';

import TriangleIcon from '@material-ui/icons/Details';
import SquareIcon from '@material-ui/icons/CheckBoxOutlineBlank';
import CircleIcon from '@material-ui/icons/Adjust';

import { extractTernaryDigits } from '../ternary';

const MIN_COLOR = '#21ff4a';
const MAX_COLOR = '#2167ff';
const MAX_POWER = 5;

const makeColor = colors(MIN_COLOR, MAX_COLOR);

const makeScale = (power) => ({
  'display': 'inline-block',
  'transform': 'scale(' + (1 + power / MAX_POWER).toFixed(1) + ')',
  'margin': (power * 3) + "px",
});

const TernaryString = ({ digits }) => {
  const ternaryDigits = extractTernaryDigits(digits);
  const symbols = [];

  ternaryDigits.forEach(d => {
    const symbol = d[0];
    const power = d[1];

    const symbolColor = makeColor(parseInt(power) / MAX_POWER);
    const icon =   (symbol === 'a' ? <CircleIcon htmlColor={symbolColor} />
                    :  symbol === 'b' ? <SquareIcon htmlColor={symbolColor} />
                    : symbol === 'c' ? <TriangleIcon htmlColor={symbolColor} />
          : null);
    symbols.push(<span key={symbols.length} style={makeScale(power)}>{ icon }</span>);
  });

  return <span>{ symbols }</span>;
};

export default TernaryString;
