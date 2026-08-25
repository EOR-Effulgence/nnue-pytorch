#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "nnue_training_data_formats.h"
#include "../YaneuraOu/source/learn/learn.h"

#include <optional>
#include <fstream>
#include <string>
#include <memory>

// ppl.h (Microsoft Parallel Patterns Library) は Windows 専用。
// Linux ビルド (クラウド学習、L-base #289) では標準 C++ のスレッドで代替する。
#ifdef _WIN32
#include <ppl.h>
#else
#include <thread>
#include <algorithm>
#endif

namespace training_data {

    using namespace binpack;

    static bool ends_with(const std::string& lhs, const std::string& end)
    {
        if (end.size() > lhs.size()) return false;

        return std::equal(end.rbegin(), end.rend(), lhs.rbegin());
    }

    static bool has_extension(const std::string& filename, const std::string& extension)
    {
        return ends_with(filename, "." + extension);
    }

    static std::string filename_with_extension(const std::string& filename, const std::string& ext)
    {
        if (ends_with(filename, ext))
        {
            return filename;
        }
        else
        {
            return filename + "." + ext;
        }
    }

    struct BasicSfenInputStream
    {
        virtual std::optional<TrainingDataEntry> next() = 0;
        virtual void fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto v = this->next();
                if (!v.has_value())
                {
                    break;
                }
                vec.emplace_back(*v);
            }
        }

        virtual bool eof() const = 0;
        virtual ~BasicSfenInputStream() {}
    };

    struct BinSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bin";

        BinSfenInputStream(std::string filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_stream(filename, openmode),
            m_filename(filename),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            Learner::PackedSfenValue e;
            bool reopenedFileOnce = false;
            for(;;)
            {
                if(m_stream.read(reinterpret_cast<char*>(&e), sizeof(Learner::PackedSfenValue)))
                {
                    auto entry = packedSfenValueToTrainingDataEntry(e);
                    if (!m_skipPredicate || !m_skipPredicate(entry))
                        return entry;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return std::nullopt;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return std::nullopt;

                        continue;
                    }

                    m_eof = true;
                    return std::nullopt;
                }
            }
        }

        void fill(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
            std::vector<Learner::PackedSfenValue> packedSfenValues(n);
            bool reopenedFileOnce = false;
            for (;;)
            {
                if (m_stream.read(reinterpret_cast<char*>(&packedSfenValues[0]), sizeof(Learner::PackedSfenValue) * n))
                {
                    vec.resize(n);
#ifdef _WIN32
                    concurrency::parallel_for(size_t(0), n, [&vec, &packedSfenValues](size_t i)
                        {
                            vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
                        });
#else
                    // Linux: std::thread で n を等分する (PPL と同じ結果、順序も保存)
                    {
                        const std::size_t hw = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
                        const std::size_t nthreads = std::min<std::size_t>(hw, n);
                        if (nthreads <= 1)
                        {
                            for (std::size_t i = 0; i < n; ++i)
                                vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
                        }
                        else
                        {
                            std::vector<std::thread> workers;
                            workers.reserve(nthreads);
                            const std::size_t chunk = (n + nthreads - 1) / nthreads;
                            for (std::size_t t = 0; t < nthreads; ++t)
                            {
                                const std::size_t begin = t * chunk;
                                const std::size_t end = std::min(begin + chunk, n);
                                if (begin >= end) break;
                                workers.emplace_back([&vec, &packedSfenValues, begin, end]()
                                    {
                                        for (std::size_t i = begin; i < end; ++i)
                                            vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
                                    });
                            }
                            for (auto& w : workers) w.join();
                        }
                    }
#endif
                    return;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return;

                        continue;
                    }

                    m_eof = true;
                    return;
                }
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinSfenInputStream() override {}

    private:
        std::fstream m_stream;
        std::string m_filename;
        bool m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));

        return nullptr;
    }

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file_parallel(int concurrency, const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        // TODO (low priority): optimize and parallelize .bin reading.
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));

        return nullptr;
    }
}

#endif
