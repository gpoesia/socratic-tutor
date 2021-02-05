const lodash = require('lodash');
const {
  guess_step_exercises,
  guess_state_exercises,
} = require('../../exercises.json');

const config = require('../../config.json');

export default (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const policy = params.policy || 'random';
  let chosen;

  if (lodash.sample(['step', 'state']) == 'step') {
    chosen = {
      'type': 'guess-step',
      'exercise': lodash.sample(guess_step_exercises),
    };
  } else {
    chosen = {
      'type': 'guess-state',
      'exercise': lodash.sample(guess_state_exercises),
    };
  }

  console.log('Chosen:', chosen)

  res.json(chosen);
};
