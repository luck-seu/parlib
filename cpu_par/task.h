#ifndef CPU_PAR_TASK_H_
#define CPU_PAR_TASK_H_

#include <functional>
#include <vector>

namespace luck::parlib::cpu {

// Alias type name for an executable function.
typedef std::function<void()> Task;

// Alias type name for an array of executable functions.
typedef std::vector<Task> TaskPackage;

}  // namespace luck::parlib::cpu

#endif  // CPU_PAR_TASK_H_
