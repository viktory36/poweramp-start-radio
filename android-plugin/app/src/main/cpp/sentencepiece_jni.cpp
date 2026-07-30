#include <jni.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sentencepiece_processor.h"

namespace {

void Throw(JNIEnv* env, const char* class_name, const std::string& message) {
    jclass exception_class = env->FindClass(class_name);
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message.c_str());
    }
}

void AppendUtf8(uint32_t code_point, std::string* output) {
    if (code_point <= 0x7f) {
        output->push_back(static_cast<char>(code_point));
    } else if (code_point <= 0x7ff) {
        output->push_back(static_cast<char>(0xc0 | (code_point >> 6)));
        output->push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    } else if (code_point <= 0xffff) {
        output->push_back(static_cast<char>(0xe0 | (code_point >> 12)));
        output->push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
        output->push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    } else {
        output->push_back(static_cast<char>(0xf0 | (code_point >> 18)));
        output->push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3f)));
        output->push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
        output->push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    }
}

std::string JavaStringToUtf8(JNIEnv* env, jstring input) {
    const jsize length = env->GetStringLength(input);
    const jchar* chars = env->GetStringChars(input, nullptr);
    if (chars == nullptr) return {};

    std::string output;
    output.reserve(static_cast<size_t>(length) * 3);
    for (jsize i = 0; i < length; ++i) {
        uint32_t code_point = chars[i];
        if (code_point >= 0xd800 && code_point <= 0xdbff && i + 1 < length) {
            const uint32_t low = chars[i + 1];
            if (low >= 0xdc00 && low <= 0xdfff) {
                code_point = 0x10000 + ((code_point - 0xd800) << 10) + (low - 0xdc00);
                ++i;
            } else {
                code_point = 0xfffd;
            }
        } else if (code_point >= 0xd800 && code_point <= 0xdfff) {
            code_point = 0xfffd;
        }
        AppendUtf8(code_point, &output);
    }
    env->ReleaseStringChars(input, chars);
    return output;
}

sentencepiece::SentencePieceProcessor* Processor(jlong handle) {
    return reinterpret_cast<sentencepiece::SentencePieceProcessor*>(handle);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_powerampstartradio_indexing_OfficialSentencePieceTokenizer_nativeCreate(
        JNIEnv* env, jobject, jstring model_path) {
    if (model_path == nullptr) {
        Throw(env, "java/lang/IllegalArgumentException", "SentencePiece model path is null");
        return 0;
    }

    const std::string path = JavaStringToUtf8(env, model_path);
    auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
    const auto status = processor->Load(path);
    if (!status.ok()) {
        Throw(env, "java/lang/IllegalArgumentException",
              "Cannot load SentencePiece model: " + status.ToString());
        return 0;
    }
    return reinterpret_cast<jlong>(processor.release());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powerampstartradio_indexing_OfficialSentencePieceTokenizer_nativeEncode(
        JNIEnv* env, jobject, jlong handle, jstring input) {
    auto* processor = Processor(handle);
    if (processor == nullptr) {
        Throw(env, "java/lang/IllegalStateException", "SentencePiece tokenizer is closed");
        return nullptr;
    }
    if (input == nullptr) {
        Throw(env, "java/lang/IllegalArgumentException", "Text query is null");
        return nullptr;
    }

    std::vector<int> ids;
    const auto status = processor->Encode(JavaStringToUtf8(env, input), &ids);
    if (!status.ok()) {
        Throw(env, "java/lang/IllegalArgumentException",
              "SentencePiece encoding failed: " + status.ToString());
        return nullptr;
    }

    jintArray result = env->NewIntArray(static_cast<jsize>(ids.size()));
    if (result == nullptr) return nullptr;
    if (!ids.empty()) {
        env->SetIntArrayRegion(result, 0, static_cast<jsize>(ids.size()), ids.data());
    }
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powerampstartradio_indexing_OfficialSentencePieceTokenizer_nativeDestroy(
        JNIEnv*, jobject, jlong handle) {
    delete Processor(handle);
}
