

# Model

## weight

モデルを huggingface からローカルにダウンロードしておく。コンテナ起動時にキャッシュディレクトリをバインドして使用する。

```shell
/home/nvidia/.local/bin/huggingface-cli download liuhaotian/llava-v1.5-13b
/home/nvidia/.local/bin/huggingface-cli download liuhaotian/llava-v1.5-7b
```


# Libraries

LLaVA の通常の方法で `pip install -e .` を入れると、pytorch も入ってしまい Jetson アーキテクチャで GPU を利用できなくなるため、`pyproject.toml` をカスタムしている。

- torch, torchvision, timm を削除
    - torch がインストールされないようにしている
- bitsandbytes の削除
    - bitsandbytes もクセがあり、Jetsonアーキテクチャでうまく入らなかったため削除