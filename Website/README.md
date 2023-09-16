# Running the express server:

## Production

In Website folder
```sh
cd client
npm run build
cd ..
PORT=6969 node .
```


## Dev

In Website folder
`PORT=6969 IGNORE_AUTH=1 node .`
In Website/client folder
`REACT_APP_HOST_API=http://localhost:6969/ npm run start`

In Another window *ONLY IF IT DOESNT WORK RIGHT WITHOUT THIS*
`ssh -p 22222 -N -L localhost:6969:localhost:6969 vthacks@movementgaming.online`
