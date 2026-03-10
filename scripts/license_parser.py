import logging
import os
from collections import defaultdict
from copy import copy

import yaml
from tabulate import tabulate

from works import Work, EVENT
from llm_license_helper import llm_helper


""" Use for standardize the string list of license rights """
def standardize_string_list(string_list):
    processed_list = []
    for string in string_list:
        processed_string = string.title().replace("_", " ")
        processed_list.append(processed_string)
    return ", ".join(processed_list)


class License(object):
    def __init__(self, license_meta, usages_dict):
        self.meta = license_meta
        self.usages_dict = usages_dict
        self.full_name = license_meta["full_name"]
        self.short_id = license_meta["short_id"]
        self.relicense = False

    def is_public(self):
        return "public" in self.meta["categories"]

    def is_public_domain(self):
        return "Public Domain" in self.meta["labels"]

    def is_copyleft(self):
        return "copyleft" in self.meta["categories"]

    def is_disclose(self):
        return "disclose" in self.meta["categories"]

    def is_auto_relicensing(self):
        return "auto-relicensing" in self.meta["categories"]

    def is_permissive(self):
        return "permissive" in self.meta["categories"]

    def is_irrevocable(self):
        return "irrevocable" in self.meta["rights_prefix"]

    def has_compat(self):
        return "compat" in self.meta

    def get_compat_list(self):
        return self.meta.get("compat") if "compat" in self.meta else None

    def get_incompat_list(self):
        return self.meta.get("incompat") if "incompat" in self.meta else None

    def is_granted_right(self, right: str):
        if self.is_public_domain():
            return True
        if right in self.meta["rights"]:
            return True
        if right in self.meta["reserved_rights"]:
            return False
        return "NODEF"

    def is_supported_work_type(self, type: str):
        if type in self.meta["categories"]:
            return True
        if "software" in self.meta["categories"] and type == "model":
            return True
        return False

    def get_share_coverage(self) -> list:
        return self.meta.get("coverage", [])

    def summary(self, save_path=None):
        notices = []
        if self.is_copyleft():
            notices.append("Copyleft")
        if not self.is_irrevocable():
            notices.append("Revocable")
        if not self.is_public():
            notices.append("Not Public")
        data = [
            ["License Name", self.full_name],
            ["License Tags", standardize_string_list(self.meta["categories"])],
            ["Granted Rights", standardize_string_list(self.meta["rights"])],
            ["Reserved Rights", standardize_string_list(self.meta["reserved_rights"])],
            ["Notices for Use", "None" if not notices else ", ".join(notices)],
            ["Full Text Link", self.meta["url"]],
        ]
        text = tabulate(data, tablefmt="plain")
        if save_path:
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
        return data


class Parser(object):
    def __init__(self, path):
        # 允许传入相对路径：自动相对于当前文件所在目录解析，避免受工作目录影响
        if not os.path.isabs(path):
            base_dir = os.path.dirname(__file__)
            path = os.path.join(base_dir, path)
        self.path = path
        try:
            with open(self.path, encoding="utf-8") as licenses_file:
                yml_content = yaml.safe_load(licenses_file)
            usages_dict, licenses_list = yml_content[0], yml_content[1:]
        except IOError:
            logging.error("Failed to open the licenses file in %s.", self.path)
            usages_dict, licenses_list = {}, []

        self.licenses_dict = {}
        for li in licenses_list:
            if "short_id" in li:
                license_meta = li
                obj = License(license_meta, usages_dict)
                self.licenses_dict[li["short_id"]] = obj
        self.licenses_list = list(self.licenses_dict.keys())

    def detect_license_version(self, license_name) -> tuple[str, bool]:
        if "-" in license_name:
            maybe_name, maybe_ver_str = license_name.rsplit("-", 1)
            if maybe_ver_str.replace(".", "").isdigit():
                return maybe_name, True
        return license_name, False

    def __find_matching_license(self, license_name, ignore_case: bool = True, approx_match: bool = True) -> tuple[str, bool]:
        matching, exact_ver = None, False
        names_list = list(self.licenses_list)
        if ignore_case:
            target = license_name.casefold()
            for name in names_list:
                if name.casefold() == target:
                    matching = name
                    exact_ver = True
                    break
        else:
            if license_name in names_list:
                matching = license_name
                exact_ver = True

        if not matching and approx_match:
            base_name, has_version = self.detect_license_version(license_name)
            if ignore_case:
                base_name = base_name.casefold()
            for name in names_list:
                license_base_name, _ = self.detect_license_version(name)
                if ignore_case:
                    if license_base_name.casefold() == base_name:
                        matching = name
                        exact_ver = not has_version
                        break
                else:
                    if license_base_name == base_name:
                        matching = name
                        exact_ver = not has_version
                        break

        return matching or "", exact_ver
    
    def print_supported_license_names(self):
        print(self.licenses_list)
        return
    
    # Register license for the work or a list of works
    def register_license(self, work: Work, relicense: bool = False):
        if isinstance(work, list):    
            licenses = [self.register_license(w) for w in work]
            return licenses

        if work.license_name == 'TBD':
            logging.warning(f"The license name of Work {work.name} is not specified")
            return None
        # 处理 Unknown 许可证，尝试通过 LLM / API 自动识别
        if isinstance(work.license_name, str) and work.license_name.lower() == "unknow":
            try:
                license_name, error = llm_helper.handle_unknown_license(work.name)
                work.license_name = license_name
                logging.info("Updated license for %s from Unknow to %s", work.name, license_name)
            except Exception as e:
                logging.error("Failed to handle unknown license for %s: %s", work.name, e)

        license = self.clone_license(work.license_name)
        if license is None:
            logging.error("Cannot find the license decription of %s in %s", work.license_name, self.path)
            work.add_event(EVENT.LICENSE_NO_FOUND_ERROR)
            return None

        if self.work_type_check(license, work) == False:
            work.add_event(EVENT.LICENSE_TYPE_MISMATCH_WARNING)
            logging.warning(f"{license.short_id} is no suitable for {work.name} ({work.type}) licensing")

        if work.is_registered() and relicense == False:
            logging.debug("%s already has %s license, ignore the license registration", work.name, license.short_id)
        else:
            work.set_license(license)
            logging.debug("%s is registered as %s", work.name, work.license_name)
        return license


    def clone_license(self, license_name: str, approx_match: bool = True) -> License | None:
        """
        根据名称查找并克隆 License；若本地不存在，则尝试调用 LLM 生成新的条目。
        """
        match_license, exact_ver = self.__find_matching_license(license_name, approx_match)
        if match_license:
            logging.debug("Find the matching license: %s, version matching: %s", match_license, exact_ver)
            clone = copy(self.licenses_dict[match_license])
            return clone

        # 本地未找到：尝试用 LLM 自动建模该许可证
        logging.info("License %s not found in local database. Attempting to analyze with LLM...", license_name)
        try:
            license_data = llm_helper.get_license_data(license_name)
            if license_data:
                # 复用任意已有 license 的 usages_dict（所有许可证共用）；若无则用与 licenses_description.yml 一致的默认
                any_license = next(iter(self.licenses_dict.values()), None)
                if any_license:
                    usages_dict = any_license.usages_dict
                else:
                    usages_dict = {
                        "copy": ["use"],
                        "use": ["use"],
                        "share": ["redistribute", "sublicense"],
                        "sell": ["redistribute", "sublicense", "commercial_use"],
                        "modify": ["modify"],
                        "train": ["use", "modify"],
                        "combine": ["use", "merge"],
                        "combine_mix": ["use", "merge"],
                        "amalgamate": ["use", "modify"],
                        "distill": ["use"],
                        "generate": ["use"],
                        "embed": ["use", "modify"],
                        "stat": ["use"],
                    }
                new_obj = License(license_data, usages_dict)
                self.licenses_dict[license_name] = new_obj
                self.licenses_list.append(license_name)
                logging.info("Successfully added %s to license database using LLM analysis", license_name)
                return copy(new_obj)
            logging.warning("Failed to analyze %s with LLM", license_name)
        except Exception as e:
            logging.error("Error while using LLM to analyze %s: %s", license_name, e)

        return None
    
    # Check whether the license can support this type of work (software, model, data). Note, work maybe unregistered.
    def work_type_check(self, license:License, work:Work) -> bool:
        if work.type == 'mix':
            if work.is_include_mixworks() == True: # Mixworks list contant at least one work
                # IF this work is mix of works, return True IF the license can match at least one of mixworks' type
                for (mixw, _) in work.mixworks:
                    if self.work_type_check(license, mixw) is True: # Recursion
                        return True
            else: # The mixworks list of this 'mix' type work is empty
                logging.warn(f"The type of {work.name} is 'mix' but its has no mixworks")
                work.add_event(EVENT.MIXWORKS_NO_FOUND_WARNING)
        elif license.is_supported_work_type(work.type) is True: # Work type is one of 'model', 'data', 'algorithm'
            return True
        #license.caution['warnings']['TYPE_MISMATCH'].append((license.short_id, type))
        return False

    # Analysis the right grantable and licensing (license name='TBD') of work
    # The supported 'open_policy' are 'sell', 'share', 'personal'
    # NOTE: The subworks are considered as be released with this work but auxworks are not.
    def analysis(self, work:Work, open_policy:str='share'):

        # 1, Find all relied and no licensed works ('TBD') for licensing analysis, include this work
        # Then, determine license for 'TBD' works by call license_analysis
        works_sequence = [w for w, _ in work.find_relied_works() if w.license_name == 'TBD'] + [work]
        for alys_work in works_sequence:
            if self.license_analysis(alys_work) is False:
                logging.error('License analysis failed, the final license type could not be determined.')
                # 在服务端场景不再强制退出整个进程，返回 False 由调用方决定如何处理
                return False
        
        # 2, Check the rights granting for reuse and open (such as share, sell)
        self.rights_granting_analysis(work, open_policy)

        # 3, Check the warnings involved by relied works
        self.warning_analysis(work, open_policy)

        # 4, Check the restrictions involved by relied works
        self.restrictions_analysis(work, open_policy)
        
        # 5, Finnal check for disclose and redistribute requirements
        self.redistribution_analysis(work, open_policy)
        return True

    """
        Register new license for TBD work, the default license is 'Unlicense'.
        If there has one copyleft license, license proliferation will be applied,
        If there has two or more different copyleft licenses, an error will be thrown if they are in-compat.
        NOTE: This function only consider the first level relied works that with explicit licenses, please loop this function if need.
    """
    def license_analysis(self, TBD_work:Work) -> bool:
        if TBD_work.license_name != 'TBD':
            logging.warning(f"{TBD_work.name} has license, no need for license analysis")
            return True
        
        elif TBD_work.has_relied_work(exclude='aux') is False: # license_name == TBD but w/o relied subworks and mixworks.
            TBD_work.license_name = 'Unlicense' # Default to 'Unlicense'
            self.register_license(TBD_work)
            return True

        # MAIN FUNCTION. license_name == TBD, this work has relied works (first level), and all these works are registered
        elif all(rw.is_registered() == True for rw, _ in TBD_work.mixworks + TBD_work.subworks + TBD_work.auxworks):
            relied_works = TBD_work.mixworks + TBD_work.subworks + TBD_work.auxworks # First level relied works
            relied_works_wo_aux = TBD_work.mixworks + TBD_work.subworks # W/O auxworks
            
            # Find applied term in each relied work
            terms = {rw: self.find_applied_term(rw, ru) for rw, ru in relied_works}
            terms_wo_aux = {rw: self.find_applied_term(rw, ru) for rw, ru in relied_works_wo_aux}

            copylefts = [rw for rw, _ in relied_works_wo_aux if rw.license.is_copyleft() == True] # Relied works (w/o aux) that under copyleft licenses
            permissives = [rw for rw, _ in relied_works_wo_aux if rw.license.is_permissive() == True] # Relied works (w/o aux) that under permissive licenses
            public_domains = [rw for rw, _ in relied_works_wo_aux if rw.license.is_public_domain() == True]
            others = [rw for rw, _ in relied_works_wo_aux if rw not in copylefts+permissives+public_domains]

            # Find triggered copylefts works that result in copyleft license proliferation 
            triggered_copylefts = [rw for rw in copylefts if terms_wo_aux[rw]['result'] not in ['independent', 'NODEF']]
            triggered_permissive = [rw for rw in permissives if terms_wo_aux[rw]['result'] not in ['independent', 'NODEF']]
            # Public domain will not be triggered in our design
            triggered_others = [rw for rw in others if terms_wo_aux[rw]['result'] not in ['independent', 'NODEF']]
            
            # Find a feasible relicense solution of this work from multiple licenses
            aim_license_name, fail_works = self.multiple_license_solver(terms_wo_aux, triggered_copylefts, triggered_permissive, public_domains, triggered_others, TBD_work.assigned_license_name, TBD_work.name)
            logging.debug(f"Aim license name: {aim_license_name}")

             # Fail to found the solution, force it to aim license and report an error
            if len(fail_works) > 0: 
                logging.debug(f"Fail to relicensing works: {[w.name for w in fail_works]}")

                fail_copyleft = [fw for fw in fail_works if fw.license.is_copyleft()]
                fail_incompat = [fw for fw in fail_works if terms_wo_aux[fw].get('relicense') == 'conditional']
                fail_noright = [fw for fw in fail_works if terms_wo_aux[fw].get('relicense') == False]

                """
                if len(fail_copylefts) >= 2:
                    # Multiple in-compat copyleft exist
                    TBD_work.add_event(EVENT.MULTIPLE_COPYLEFT_ERROR(fail_copylefts))
                """
                for w in fail_incompat:
                    TBD_work.add_event(EVENT.LICENSE_IN_COMPAT_ERROR(w.name, w.license_name, aim_license_name))
                    logging.error(f'Work {w.name} under {w.license_name} cannot be relicensed to {aim_license_name} due to they are in-compat.')
                for w in fail_noright:
                    TBD_work.add_event(EVENT.RIGHT_NO_GRANT_ERROR(w.name, 'relicense'))
                    logging.error(f'Work {TBD_work.name} canot be licensed as {aim_license_name} due to {w.name} does not grant the relicense rights')

            # Despite exist fails, we report errors and force relicense the result as aim license
            prev_license_name = TBD_work.license_name
            TBD_work.license_name = aim_license_name # Set to new name
            self.register_license(TBD_work, relicense=True) # Relicensing, register the new license!
            logging.debug(f'Work {TBD_work.name} be relicensed from {prev_license_name} to {TBD_work.license_name}')
            if TBD_work.license_name == TBD_work.assigned_license_name:
                logging.debug(f"Work {TBD_work.name} was successfully licensed as {TBD_work.assigned_license_name}")
                TBD_work.assigned_license_name = None # Reset assignment

            return True # Always return True, just report errors
        
        return False

    """
        Return True if the original license can be relicensed to new license according to its term.
    """
    def relicense_solver(self, term, original_license_name, new_license_name) -> bool:
        # If original license is public domain license, or the original license is same as new license, always return true
        oli = self.clone_license(original_license_name, approx_match=False)
        nli = self.clone_license(new_license_name, approx_match=False)
        if oli is None or nli is None: # Cannot find the matching license in parser, return False
            logging.debug('Cannot complete the relicense solver due to the license name no found')
            return False

        if oli.short_id == nli.short_id or oli.is_public_domain(): # Always return ture if original license is same as new licenses, or original license is a public domain license
            return True
        # Relicensing is permitted if the resulted product is independent or NODEF
        if term.get('result') in ['independent', 'NODEF']:
            return True
        
        def compat_matching(liname, compat_list): # if compat=[GPL], any version of GPL is compat, if compat=[GPL-3.0], only GPL-3.0 can be matched
            matching = None
            if compat_list is None: return matching
            for compat_name in compat_list:
                cname, hasver =  self.detect_license_version(compat_name)
                if hasver and liname == cname:
                    matching = compat_name # Return the first matching license name
                    break
                if not hasver and self.detect_license_version(liname)[0] == cname: # No need comapre the version of the new license
                    matching = compat_name
                    break
            return matching

        # If relicense if conditional, check the compatibility between these licenses
        if term.get('relicense', '') == 'conditional': # Check the compatibility of licenses, ignore version number
            original_compat_list, original_incompat_list = oli.get_compat_list(), oli.get_incompat_list()
            has_compat = False if oli.get_compat_list() is None else True
            has_incompat = False if oli.get_incompat_list() is None else True
            is_compat, is_incompat =  compat_matching(new_license_name, original_compat_list), compat_matching(new_license_name, original_incompat_list)

            if has_compat and not has_incompat: # Only the license in compat list are allowed
                return True if is_compat else False
            elif has_incompat and not has_compat: # Only the license in in-compat are prohibited
                return False if is_incompat else True
            elif has_compat and has_incompat: # Both privide compat list and incompat list, Priority: compat > incompat
                logging.warning(f"The relicense term of {original_license_name} has compat list and in-compat list.")
                if is_compat: return True
                if is_incompat: return False
                return False
            else: # Not provide compat list and incompat list
                logging.error(f"The relicense term of {original_license_name} is conditional but not compat list and in-compat list are provided.")
                return False

        return term.get('relicense', False) # Can or Cannot relicense, no condition.
        
    """
        Find a feasible relicense solution (aim license) for multiple license
        Also return a list of fail works
    """
    def multiple_license_solver(self, terms, copylefts, permissives, public_domains, others, aim_license_name = None, work_name = 'NA') -> tuple[str, list]:
        check_relicense_works = permissives + copylefts 
        fail_works = []
        
        # No licenses need to be check
        if len(check_relicense_works) == 0: 
            if aim_license_name:
                return aim_license_name, []
            else:
                return 'Unlicense', []
            
        # We intend to fullfill the prior license assignment of this work if it is exist
        if aim_license_name:
            # Try to fufill the aim license be assigned to this work, check the relicensable of triggered copylefts and all permissives
            relicensable = [self.relicense_solver(terms[w], w.license.short_id, aim_license_name) for w in check_relicense_works]
            if all(relicensable):
                logging.debug(f"Can fullfill the license assignment of work {work_name} by relicensing its relied works")
                return aim_license_name, []
            else:
                # relicense_pass = False # NOTE: we dont set this value here because we will find another aim license later
                fail_works = [w for w, x in zip(check_relicense_works, relicensable) if x == False]
                logging.error(f"Cannot fullfill the license assignment of work {work_name} due to the in-compat of {fail_works}") 
        
        if len(copylefts) > 0:
            """ #1: w/o copyleft
            Find a feasible solution if the aim license cannot be achieved or there is no assigned aim license
            We consider the situtaion that contain triggered copyleft first, their proliferation should be handled.
            """

            for tcl in copylefts: # Seach feasible solution from each triggred copyleft
                if all(self.relicense_solver(terms[w], w.license.short_id, tcl.license.short_id) for w in check_relicense_works):
                    aim_license_name = tcl.license_name
                    break
            if aim_license_name is None: # Not any feasible copyleft found, default to the first copyleft 
                aim_license_name = copylefts[0].license.short_id
                logging.debug(f"Cannot solve the compatibility within copyleft and aim license, change the aim license to {aim_license_name} instead.")

        elif len(permissives) > 0:
            """ #2: w/o copyleft w/ permissive situtation
                no license proliferation, need check compatibility
            """

            # Prefer to use Unlicense
            candidate_permissive_name = ['Unlicense'] + [w.license.short_id for w in check_relicense_works if w.license.short_id != 'Unlicense']
            for cpn in candidate_permissive_name:
                if all(self.relicense_solver(terms[w], w.license.short_id, cpn) for w in check_relicense_works):
                    aim_license_name = cpn
                    break
            if aim_license_name is None: # Not any feasible permissive found, default to the first permissive
                aim_license_name = permissives[0].license.short_id
                logging.debug(f"Cannot solve the compatibility within permissive, change the aim license to {aim_license_name} instead.")

        elif len(public_domains) > 0 and len(others) == 0:
            """ #2: All works are under public domain licenses
                    Always allow relicense, default to Unlicense
            """
            aim_license_name = 'Unlicense'
        else:
            logging.warning(f"Unexpect situtation: {[w.license_name for w in others]}")
            aim_license_name = 'Unlicense'
            return aim_license_name, others

        # Recheck the relicenable of works, collect the fail to relicense works
        relicensable = [self.relicense_solver(terms[w], w.license.short_id, aim_license_name) for w in check_relicense_works]
        fail_works = [w for w, x in zip(check_relicense_works, relicensable) if x == False]

        return aim_license_name, fail_works

    # Find applied terms for this work
    def find_applied_term(self, work:Work, usage:str) -> dict:
        terms = work.license.meta['terms']
        for t in terms:
            # Matching requirements: 1) usage is in definition; 2) work form is also in definition.
            if usage in t['usages'] and work.form in t['forms']:
                return t # Return the first found term
        for t in terms: 
            if usage.split('_')[0] in t['usages'] and work.form in t['forms']:
                return t # Return match like 'combine_mix' -> 'combine' if only 'combine' has been defined
        return defaultdict(list)

    # Analysis rights granting
    def rights_granting_analysis(self, work:Work, open_policy:str):
        # Deal with share policy
        shared_works_list = [w for w, u in work.find_shared_works()]
        req_open_policy_rights = []
        if open_policy == 'personal': # No share and redistribute this work
            work.add_event(EVENT.PERSONAL_OPEN_POLICY)
            logging.debug(f"Work {work.name} will not be shared and does not require any permissions")
        elif open_policy in ['sell', 'share']:
            work.add_event(getattr(EVENT, (open_policy + '_OPEN_POLICY').upper()))
            req_open_policy_rights = work.license.usages_dict[open_policy]

        for rw, ru in work.find_relied_works(): 
            # Check whether the required rights is granted
            req_rights = copy(work.license.usages_dict[ru])
            if rw in shared_works_list:
                req_rights += req_open_policy_rights
            
            for right in req_rights:
                is_granted = rw.license.is_granted_right(right)
                if is_granted == False:
                    if right == 'sublicense' and rw.license.is_auto_relicensing() == True:
                        continue # Skip the check of sublicense right if this work has an auto-relicesing mechanism
                    work.add_event(EVENT.RIGHT_NO_GRANT_ERROR(rw.name, right))
                    logging.error(f"The required {right} right is not granted by Work {rw.name}")
                elif is_granted == 'NODEF':
                    # 若许可证未明确授予该权利，则统一记为通用的 RIGHT_NO_GRANT_WARNING 事件
                    work.add_event(EVENT.RIGHT_NO_GRANT_WARNING(rw.name, right))
                    logging.warn(f"The required {right} right is not explicitly granted by Work {rw.name}")
        return

    def restrictions_analysis(self, work:Work, open_policy:str=''): # open_plicy -> no use
        for rw, ru in work.find_relied_works(): 
            # Gather corresponding terms for this work
            term = self.find_applied_term(rw, ru)
            if 'restrictions' in term:
                for res in term.get('restrictions'):
                    work.add_event(getattr(EVENT, res.upper())(rw.name, rw.license_name))
        return
    
    def warning_analysis(self, work:Work, open_policy:str=''):
        for rw, ru in work.find_relied_works(exclude=['aux']): # Dont report warnings from auxworks
            warnings = rw.filter_events_by_type('warning')
            #print(rw.name, rw.caution)
            if len(warnings) > 0:
                work.add_event(warnings)
        return
    
    def redistribution_analysis(self, work:Work, open_policy:str):
        # Add the redistribution requirements of the relied works and this work if it will be shared
        if open_policy in ['share', 'sell']:
            for w, u in work.find_shared_works() + [(work, 'copy')]: # This 'work' be shared as copy
                term = self.find_applied_term(w, u)
                
                if w.license.is_disclose() == True and w.form != 'raw': # Disclose Source Check, only check for shared works
                    work.add_event(EVENT.LICENSE_DISCLOSE_SELF_WARNING(w.name, w.license_name))
                
                if term['result']:
                    if term['result'] not in (['independent', 'NODEF'] + w.license.get_share_coverage()):
                        work.add_event(EVENT.LICENSE_SHARING_PROHIBITED_ERROR(w.name, w.license_name)) # This work canot be shared according its license
                    else:
                        for req in w.license.meta['redistribute']:
                            work.add_event(getattr(EVENT, req.upper())(w.name, w.license_name)) # The redistribution requreiments from all shared works (e.g. mixworks and subworks)
            
        # Disclose Source Check
        '''
        for w in [rw for rw, _ in work.find_relied_works()] + [work]:
            if w.license.is_disclose() == True and w.form != 'raw':
                work.add_event(EVENT.LICENSE_DISCLOSE_SELF_WARNING(w.name, w.license_name))
        '''
        return