const { UserSession } = require('../../lib/data');
const uuid = require('uuid');

export default async (req, res) => {
  const newSessionId = uuid.v4();

  const session = new UserSession({ id: newSessionId });
  session.beginTimestamp = new Date();
  session.testResponses = [];
  await session.save();

  res.send({ id: newSessionId });
}
