const _ = require('lodash');
const { UserSession } = require('../../lib/data');
const {
  guess_step_exercises,
  guess_state_exercises,
} = require('../../exercises.json');

guess_step_exercises.forEach((p, id) => p.id = id);
guess_state_exercises.forEach((p, id) => p.id = id);

const MIN_SUCCESSES = 5;

const config = require('../../config.json');

export default async (req, res) => {
  const params = JSON.parse(req.query.params || '{}');
  const sessionId = params.id;
  console.log('Session ID:', sessionId);

  const session = await UserSession.findOne({ id: sessionId });

  console.log(session.exerciseResponses.length, 'exercise responses.');
  const correct = _.sum(session.exerciseResponses.map(r => (r.response == 0)));
  console.log(correct, 'correct responses.');

  if (correct >= MIN_SUCCESSES) {
    return res.json({ problem: null, progress: 1.0, done: true });
  }

  const policy = params.policy || 'random';
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

  res.json({ problem: chosen, progress: correct / MIN_SUCCESSES, done: false });
};
