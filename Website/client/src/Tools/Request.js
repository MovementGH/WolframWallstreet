import $ from 'jquery';

function HTTP(Method,URL,Body,Headers) {
    return new Promise((Resolve,Reject)=>{
        $.ajax({
            type: Method,
            url: URL,
            contentType: 'application/json',
            data: JSON.stringify(Body),
            headers: Headers,
            success: (Data,Status) => Resolve({Body: Data, Status: Status}),
            error: (Request) => Reject({Body: Request.responseText, Status: Request.status})
        })
    });
}

function NonCacheableRequest(Method,Endpoint,Body) {
    return HTTP(Method,(process.env.REACT_APP_HOST_API || '')+'api/'+Endpoint,Body);
}

let APICache=[];
async function CacheableRequest(Method,Endpoint,Body) {
    let Result=APICache.filter(Entry=>JSON.stringify(Entry.Body)===JSON.stringify(Body)&&Entry.Endpoint===Endpoint&&Entry.Method===Method)[0];
    if(Result) {
        return Result.Result ? Result.Result : await Result.Promise;
    }
    else {
        let Request = {
            Endpoint: Endpoint,
            Body: Body,
            Method: Method,
            Promise: NonCacheableRequest(Method,Endpoint,Body)
        };
        APICache.push(Request);
        Request.Result=await Request.Promise;
        return Request.Result;
    }
}
async function UncacheRequest(Method,Endpoint,Body) {
    APICache=APICache.filter(Entry=>JSON.stringify(Entry.Body)!==JSON.stringify(Body)||Entry.Endpoint!==Endpoint||Entry.Method!==Method);
}


const Export = {
    HTTP,
    Request: NonCacheableRequest,
    CacheableRequest,
    UncacheRequest,
}

export default Export;