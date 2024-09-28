# 简易大模型推理系统

## 实现功能

1. 文本生成
2. 多轮对话
3. 对话切换
4. 历史对话查看
5. 历史对话撤销重推理

## 文本生成

![image-20240928114358547](./README.assets/image-20240928114358547.png)

## 多轮对话

1. 输入界面![image-20240928114512163](./README.assets/image-20240928114512163.png)

2. 多轮对话实现![image-20240928115309046](./README.assets/image-20240928115309046.png)

3. 历史对话查看

   输入 `history` 查看历史对话

   ![image-20240928115351672](./README.assets/image-20240928115351672.png)

4. 历史对话撤销重推理

   输入 `regenerate` 将历史对话撤销重推理

   ![image-20240928115725733](./README.assets/image-20240928115725733.png)
   再次输入 `history` 此时原历史对话已撤销，并修改为新推理内容

   ![image-20240928120404922](./README.assets/image-20240928120404922.png)

5. 新对话创建

   输入 `new chat` 即可创建新对话，`chat-id` 依次增加

   ![image-20240928115823792](./README.assets/image-20240928115823792.png)

6. 对话切换

   输入 `chat chat-id`，即可实现对话之间的切换

   ![image-20240928115841522](./README.assets/image-20240928115841522.png)

7. 退出

   输入 `exit` 退出程序

   ![image-20240928115931319](./README.assets/image-20240928115931319.png)

