var AWS = require('aws-sdk');
var region = 'us-east-1';
var aws_access_key_id = 'AKIAJL7S3UXW2JSK5AEQ';
var aws_secret_access_key = '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';

AWS.config = {
    "accessKeyId": aws_access_key_id,
    "secretAccessKey": aws_secret_access_key,
    "region": region,
    "sslEnabled": 'true'
};

var endpoint = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com';

// Uncomment this line to use in production
// endpoint = 'https://mturk-requester.us-east-1.amazonaws.com';

var mturk = new AWS.MTurk({ endpoint: endpoint });

// This will return $10,000.00 in the MTurk Developer Sandbox
mturk.getAccountBalance(function(err, data){
  if (err) {
    console.error(err);
  } else {
    console.log(data.AvailableBalance);
  }
});
