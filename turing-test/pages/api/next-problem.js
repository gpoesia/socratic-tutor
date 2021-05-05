const Fs = require('fs');
const _ = require('lodash');
const { UserSession } = require('../../lib/data');

const config = require('../../config.json');

const {
  guess_step_exercises,
  guess_state_exercises,
} = JSON.parse(Fs.readFileSync(config['exercises']));

guess_step_exercises.forEach((p, id) => p.id = id);
guess_state_exercises.forEach((p, id) => p.id = id);

export default async (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const sessionId = params.id;
  const minimumCorrect = config['minimumCorrect'];

  const session = await UserSession.findOne({ id: sessionId });
  const correct = _.sum(session.exerciseResponses.map(r => (r.response == 0)));
  console.log(session.exerciseResponses.length, 'exercise responses,', correct);
  console.log('MIN_SUCCESSES:', minimumCorrect);
  console.log('Config:', config);

  if (correct >= minimumCorrect) {
    return res.json({ problem: null, progress: 1.0, done: true });
  }

  let chosen;

  if (_.sample(['step', 'state']) == 'step') {
    const p = _.sample(guess_step_exercises);
    chosen = {
      'type': 'guess-step',
      'exercise': p,
      'id': 'guess-step-' + p.id,
    };
  } else {
    const p = _.sample(guess_state_exercises);
    chosen = {
      'type': 'guess-state',
      'exercise': p,
      'id': 'guess-state-' + p.id,
    };
  }

  res.json({ problem: chosen, progress: correct / minimumCorrect, done: false });
};
