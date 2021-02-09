import React from 'react';

import useStore from '../lib/state';

const End = () => {
  const id = useStore(state => state.id);

  return (
    <div className="content">
      <h1>Thank you!</h1>
      <p>
        This is the end of the experiment. Thank you for your participation.
        To claim your reward on Mechanical Turk, copy the code below and
        paste it on the HIT survey:
      </p>
      <pre>{ id }</pre>
      <p>
        You may now close this tab. Have a good rest of your day!
      </p>
    </div>
  );
};
export default End;
