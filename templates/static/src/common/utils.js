/**
 * Created by root on 5/12/18.
 */


function ajaxDispatch(type, url, cfg, callBack) {
    var ajaxObj = {type: type, url: encodeURI(url), dataType: 'JSON', cache: false}, dataObj;
    if (type === 'POST') {
        if (cfg && cfg.dataObj) {
            dataObj = cfg.dataObj;
            if(dataObj.hasOwnProperty('async')){
                ajaxObj.async = false;
                delete dataObj.async;
            }
            ajaxObj.data = JSON.stringify(dataObj);
            ajaxObj.processData = false;
        }
    }
    $.ajax(ajaxObj).always(function (data, status, info) {
        if (data.status !== 'ok') {
            alert('server error: '+ data.data.status);
            return;
        }
        if (callBack) {
            callBack(data);
        }
    })
}