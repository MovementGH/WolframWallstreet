const LibExpress = require('express');
const LibCors=require('cors');
const LibBodyParser=require('body-parser');

//TODO: Proper Auth
let Users = [
    {
        id: 1,
        username: 'Admin',
    },
];
//TODO

console.log('Starting Trading server...');

function onStart() {
    console.log("Started Trading Server!");
}

function Authenticate(Request, Response, Next) {
    //TODO: Implement Accounts
    Request.user = 1; 
    Next();
}

function GetUser(Request,Response) {
    let User = Users.find(User=>User.id==Request.user);
    if(!User)
        return Response.sendStatus(404);
    return Response.json(User);
}


const ObjExpressServer=new LibExpress();
ObjExpressServer.use(LibCors({origin:'*'}));
ObjExpressServer.use(LibBodyParser.json());
ObjExpressServer.use(LibBodyParser.urlencoded({extended:true}));
ObjExpressServer.use('/', LibExpress.static('client/build'));

ObjExpressServer.use('/api', Authenticate);
ObjExpressServer.get('/api/user', GetUser);

ObjExpressServer.listen(process.env.PORT,onStart);