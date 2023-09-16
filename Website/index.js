const LibExpress = require('express');
const LibCors=require('cors');
const LibBodyParser=require('body-parser');
const LibAuth0 = require('express-openid-connect');

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

function Authenticate(Request, Response, Next) {
    if(process.env.IGNORE_AUTH=='1')
        Request.oidc={user:{nickname:'Dev User',sid:'0'}};
    else if(!Request.oidc.isAuthenticated())
        Response.redirect('/login');
    Next();
}

function GetUser(Request,Response) {
    Response.json(Request?.oidc?.user || {});
}



const ObjExpressServer=new LibExpress();
ObjExpressServer.use(LibCors({origin:'*'}));
ObjExpressServer.use(LibBodyParser.json());
ObjExpressServer.use(LibBodyParser.urlencoded({extended:true}));
ObjExpressServer.use(LibAuth0.auth(ObjAuth0Config));
ObjExpressServer.use('/', LibExpress.static('client/build'));

ObjExpressServer.use('/api', Authenticate);
ObjExpressServer.get('/api/user', GetUser);

ObjExpressServer.listen(process.env.PORT,onStart);