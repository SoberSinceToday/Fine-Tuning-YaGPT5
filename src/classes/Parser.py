from pyrogram import Client


class Parser:
    def __init__(self, api_id, api_hash) -> None:
        my_api_id = api_id
        my_api_hash = api_hash
        self.client = Client(name='userSim_session', api_id=my_api_id, api_hash=my_api_hash)

    async def parse_data(self, chat_id: str | int) -> list:
        """
        :param chat_id: username или chat_id 
        :return: список со всеми сообщениями
        """
        result = []
        async with self.client:
            async for msg in self.client.get_chat_history(chat_id, limit=100000):
                if msg.text:
                    result.append({"date": msg.date, "user": msg.from_user.id, "data": msg.text})
                # elif msg.sticker:
                #     result.append({"date": msg.date, "user": msg.from_user.username, "data": msg.sticker.file_id})
            return result
