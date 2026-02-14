# DevProc

## DevProc是什么?
DevProc是面向端侧部署场景的DSL+Compiler，覆盖前处理+torch 模型+后处理。用户在DSL中描述模型的前后处理逻辑，以及torch导出来的模型，DevProc会自动编译出目标硬件的可执行文件。
预期中的使用例子：
### 非LLM模型
``` python

import devproc
# @devproc.kernel 定义一个 pure kernel
@devproc.kernel
def preproc(prompt: devproc.String, tokenizer: devproc.Tokenizer):
    token = devproc.tokenize_encode(prompt, tokenizer)
    token = devproc.to(token, devproc.Int32)
    return token

@devproc.kernel
def postproc(logits: devproc.Tensor):
    return devproc.argmax(logits, dim=1)


@devproc.kernel
def model(token: devproc.Tensor, torch_model: devproc.TorchModel):
    return torch_model(token)

@devproc.kernel
def pipeline(prompt: devproc.String, tokenizer: devproc.Tokenizer, torch_model: devproc.TorchModel):
    token = preproc(prompt, tokenizer)
    logits = model(token, torch_model)
    return postproc(logits)

torch_model = devproc.load_torch_model("model.pt")

prompt = devproc.input("prompt", devproc.String)
tokenizer = devproc.load_tokenizer("tokenizer.json")
# jit运行，第一次运行会触发jit编译：trace → compile → cache → run。后续运行会直接从cache中获取编译结果。
out = pipeline(prompt, tokenizer, torch_model)

# aot
compiled = devproc.compile(pipeline, (prompt, tokenizer, torch_model))

# save
compiled.save("pipeline")

# load and run
lib = devproc.load("pipeline")

rt = devproc.runtime.Runtime(lib)
output = rt.run(img, img_size)
```

### LLM模型
由于LLM模型需要decode，所有使用方式会有差异。
``` python
import devproc
@devproc.kernel
def vision_preproc(img_path: devproc.String):
    img = devproc.load_image(img_path)
    img = devproc.resize(img, (224, 224))
    img = devproc.to(img, devproc.Float32)
    return img

@devproc.kernel
def vision_model(img: devproc.Tensor, torch_model: devproc.TorchModel):
    return torch_model(img)

@devproc.kernel
def vision_postproc(img: devproc.Tensor):
    img = devproc.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = devproc.transpose(img, (2, 0, 1))
    img = devproc.to(img, devproc.Float32)
    return img


@devproc.kernel
def llm_preproc(prompt: devproc.String, tokenizer: devproc.Tokenizer):
    token = devproc.tokenize_encode(prompt, tokenizer)
    token = devproc.to(token, devproc.Int32)
    return token

@devproc.kernel
def llm_tokenize_decode(tokens: devproc.Tensor, tokenizer: devproc.Tokenizer):
    token = devproc.tokenize_decode(tokens, tokenizer)
    token = devproc.to(token, devproc.Int32)
    return token

@devproc.kernel
def llm_embed(tokens: devproc.Tensor, vit_embed: devproc.Tensor, llm_embed_model: devproc.TorchModel):
    llm_embed = llm_embed_model(tokens)
    llm_embed = devproc.concat((vit_embed, llm_embed), dim=1)
    return llm_embed

@devproc.kernel
def prefill_model(torch_model: devproc.TorchModel, embeds: devproc.Tensor, kvcache: devproc.KVCache):
    logits = devproc.prefill(torch_model, embeds, kvcache)
    return devproc.argmax(logits, dim=1)

@devproc.kernel
def prefill_model(torch_model: devproc.TorchModel, embeds: devproc.Tensor, kvcache: devproc.KVCache):
    return devproc.decode(torch_model, embeds, kvcache)


@devproc.kernel
def generate(prompt: devproc.String, tokenizer: devproc.Tokenizer, torch_model: devproc.TorchModel, img_path: devproc.String):
    img = vision_preproc(img_path)
    vit_embed = vision_model(img, torch_model)
    token = llm_preproc(prompt, tokenizer)
    embeds = llm_embed(token, vit_embed, torch_model)
    kvcache = devproc.KVCache()
    prefill_token = prefill_model(torch_model, embeds, kvcache)
    decode_token = devproc.decode_loop(torch_model, prefill_token, kvcache)
    decode_token = llm_tokenize_decode(decode_token, tokenizer)
    return decode_token
```