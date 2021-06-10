#coding: utf-8
import json
import requests
from typing import Optional

from .utils._config import SLACK_WEBHOOK_URL
from .utils._config import GAS_WEBAPP_URL

def ask(text:str, 
        username:Optional[str]=None, 
        icon_url:Optional[str]=None, icon_emoji:Optional[str]=None, 
        fallback:Optional[str]=None, pretext:Optional[str]=None, 
        attachment_text:Optional[str]=None, color:str="good",
        fields_title:str="", fields_value:str="", fields_short:bool=True, 
        webhook_url:Optional[str]=None) -> requests.Response:
    """Send a question anonymously to Author's Slack using `Incoming Webhook <https://slack.com/help/articles/115005265063-Incoming-webhooks-for-Slack>`_

    Args:
        text (str)                                : Message.
        username (Optional[str], optional)        : User name. Defaults to ``None`` .
        icon_url (Optional[str], optional)        : Image url for Bot Icon. Defaults to ``None`` .
        icon_emoji (Optional[str], optional)      : Emoji for Bot Icon. Defaults to ``None`` .
        fallback (Optional[str], optional)        : A brief description of the attachment. Defaults to ``None`` .
        pretext (Optional[str], optional)         : Optional text that appears above the formatted data. Defaults to ``None`` .
        attachment_text (Optional[str], optional) : Optional text displayed in the attachment. Defaults to ``None`` .
        color (str, optional)                     : You can choose from ``"good"`` , ``"warning"``, ``"danger"``, or specify a hexadecimal color code. Defaults to ``"good"`` .
        fields_title (str, optional)              : Title of the fields. Defaults to ``""`` .
        fields_value (str, optional)              : Text value of the fields. Defaults to ``""`` .
        fields_short (bool, optional)             : Whether ``fields_value`` is short enough when displayed with other values. Defaults to ``True`` .
        webhook_url (str, optional)               : Where to send the JSON payload. Defaults to ``SLACK_WEBHOOK_URL`` .

    Returns:
        requests.Response: A server's response to an HTTP request.

    Examples:
        >>> from teilab.utils import ask
        >>> ask(text="", username=":thinking_face:", icon_emoji=":thinking_face:")
    """
    if webhook_url is None:
        ret = requests.post(url=GAS_WEBAPP_URL, data={"password": "slackwebhook"})
        webhook_url = ret.json()["dataURL"]

    fields = []
    if (len(fields_title)+len(fields_value))>0:
        fields.append({
            "title": fields_title,
            "value": fields_value,
            "short": fields_short,            
        })

    return requests.post(
        url=webhook_url,
        data=json.dumps({
            "text": text,
            "username": username,
            "icon_url": icon_url,
            "icon_emoji": icon_emoji,
            "attachments": [{
                "fallback": fallback,
                "pretext": pretext,
                "text": attachment_text,
                "color": color,
                "fields": fields,
            }]
        })
    )