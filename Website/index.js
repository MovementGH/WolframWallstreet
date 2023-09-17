const LibExpress = require('express');
const LibCors=require('cors');
const LibBodyParser=require('body-parser');
const LibAuth0 = require('express-openid-connect');
const LibChildProcess = require('child_process');
const LibFileSys = require('fs');

const Simulate = require('./simulate.js');

const ObjAuth0Config = {
  authRequired: false,
  auth0Logout: true,
  secret: 'goodsecret',
  baseURL: 'https://wolframwallstreet.movementgaming.online',
  clientID: 'SOB6qM5Zot3PQrxJS3bLfKSugGVZU7OD',
  issuerBaseURL: 'https://dev-5p5l3c47fdmmh3hz.us.auth0.com'
};

console.log('Starting Trading server...');

function onStart() {
    console.log("Started Trading Server!");
}

async function Bash(Script) {
    return new Promise((resolve, reject) => {
        let Result = '';
        LibChildProcess.exec(Script,
            (error, stdout, stderr) => {
                Result += stdout;
                Result += stderr;
                if (error !== null)
                    reject(error);
                else
                    resolve(Result);
        });
    });
}

function Authenticate(Request, Response, Next) {
    if(process.env.IGNORE_AUTH=='1')
        Request.oidc={user:{nickname:'Dev User',sid:'0'}};
    else if(!Request.oidc.isAuthenticated())
        return Response.sendStatus(401);
    Next();
}

function GetUser(Request,Response) {
    Response.json(Request?.oidc?.user || {});
}

function RunModel(Request,Response) {
    //Simulate a model

    //Get the data
    LibFileSys.readFile('../Model/output/'+Request.params.model+'.csv', (Error, Data) =>{
        let Rows = Data.toString().split('\n').slice(1,-1).map(Row=>{
            let Cells = Row.split(',');
            return {
                Date: Cells[0].split(' ')[0],
                Actual: +Cells[1],
                Predicted: +Cells[3]
            };
        });

        let Stats = Simulate(Rows, 10000, .5, 2500, 100000);
        console.log(Stats);
        Stats.SharePriceRaw = Rows.map(Row => Row.Actual);
        Stats.DatesRaw = Rows.map(Row=>Row.Date);
        Response.json(Stats);
    });

}



const ObjExpressServer=new LibExpress();
ObjExpressServer.use(LibCors({origin:'*'}));
ObjExpressServer.use(LibBodyParser.json());
ObjExpressServer.use(LibBodyParser.urlencoded({extended:true}));
ObjExpressServer.use(LibAuth0.auth(ObjAuth0Config));
ObjExpressServer.use('/', LibExpress.static('client/build'));

ObjExpressServer.use('/api', Authenticate);
ObjExpressServer.get('/api/user', GetUser);

ObjExpressServer.get('/api/model/:model/run', RunModel);

ObjExpressServer.listen(process.env.PORT,onStart);