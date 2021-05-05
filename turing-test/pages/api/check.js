const { tutorRequest } = require('../../lib/tutor.js');

export default async (req, res) => {
  const tres = await tutorRequest('check', JSON.parse(req.query.params));
  res.send(JSON.stringify(tres));
};
