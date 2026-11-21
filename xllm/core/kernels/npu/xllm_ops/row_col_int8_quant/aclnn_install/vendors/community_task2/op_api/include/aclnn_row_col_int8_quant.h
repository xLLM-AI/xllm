
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ROW_COL_INT8QUANT_H_
#define ACLNN_ROW_COL_INT8QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRowColInt8QuantGetWorkspaceSize
 * parameters :
 * x : required
 * qRowOut : required
 * qColOut : required
 * rowAbsmaxOut : required
 * colAbsmaxOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRowColInt8QuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *qRowOut,
    const aclTensor *qColOut,
    const aclTensor *rowAbsmaxOut,
    const aclTensor *colAbsmaxOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRowColInt8Quant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRowColInt8Quant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
