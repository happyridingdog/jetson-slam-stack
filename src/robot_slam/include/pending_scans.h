#pragma once
#include <cstddef>

namespace robot { namespace slam {
// Called under the input-buffer mutex. A scan selected while waiting for IMU
// must be invalidated if trimming removes that scan; otherwise syncData would
// process its old pointer but pop a different, newer scan from the queue.
template<class Clouds, class Stamps>
std::size_t trim_pending_scans(Clouds & clouds, Stamps & stamps,
    std::size_t limit, bool & selected)
{
    std::size_t skipped = 0;
    while (clouds.size() > limit) {
        clouds.pop_front();
        stamps.pop_front();
        selected = false;
        ++skipped;
    }
    return skipped;
}
}}
