

# Model

## weight

モデルを huggingface からローカルにダウンロードしておく。コンテナ起動時にキャッシュディレクトリをバインドして使用する。

```shell
/home/nvidia/.local/bin/huggingface-cli download liuhaotian/llava-v1.5-13b
/home/nvidia/.local/bin/huggingface-cli download liuhaotian/llava-v1.5-7b
```