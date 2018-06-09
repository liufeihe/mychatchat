/**
 * Created by root on 5/12/18.
 */

$(document).ready(function () {
    function sendMsg(txt) {
        ajaxDispatch('POST', '/chat', {dataObj:{msg: txt}}, function (data) {
            var dataObj = data.data;
            setMsgBox(dataObj.msg+','+dataObj.count, 'ai');
        });
    }
    function setMsgBox(msg, type) {
        var messages = $('#message-box').find('>div.messages'), msgItem='';
        if (type === 'me') {
            msgItem = '<p class="msg-item me"><span class="msg">'+msg+'</span><span class="name">: Me</span></p>'
        } else if (type === 'ai') {
            msgItem = '<p class="msg-item ai"><span class="name">AI :</span><span class="msg">'+msg+'</span></p>'
        }
        messages.append(msgItem);
        messages.scrollTop(messages.scrollTop()+30);
    }
    function bindClick() {
        $('button.send').click(function () {
            var msgDiv = $('#message-to-send'),
                txt = msgDiv.val();
            if (!txt) {
                alert('msg should not be empty.');
                return;
            }
            sendMsg(txt);
            setMsgBox(txt, 'me');
            msgDiv.val('');
        });
    }

    bindClick();
});