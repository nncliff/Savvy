import { Page } from "puppeteer";
import logger from "@karakeep/shared/logger";

/**
 * Handle common popup windows by attempting various dismissal strategies
 */
export async function handlePopups(page: Page, jobId: string): Promise<void> {
  try {
    // Strategy 1: Press ESC key to close modal/popup
    await pressEscapeKey(page, jobId);
    
    // Strategy 2: Click on empty space/overlay to close popup
    await clickEmptySpace(page, jobId);
    
    // Strategy 3: Handle specific popup types (login modals, cookie banners, etc.)
    await handleSpecificPopups(page, jobId);
    
  } catch (error) {
    logger.warn(`[Crawler][${jobId}] Error handling popups: ${error}`);
  }
}

/**
 * Press ESC key to close modal dialogs
 */
async function pressEscapeKey(page: Page, jobId: string): Promise<void> {
  try {
    // Multiple ESC key presses for stubborn popups
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);
    await page.keyboard.press('Escape');
    
    logger.debug(`[Crawler][${jobId}] Pressed ESC key multiple times to close popups`);
    
    // Wait for popups to close
    await page.waitForTimeout(800);
  } catch (error) {
    logger.debug(`[Crawler][${jobId}] ESC key press failed: ${error}`);
  }
}

/**
 * Click on empty space or overlay to close popup
 */
async function clickEmptySpace(page: Page, jobId: string): Promise<void> {
  try {
    // Try to click on common overlay/backdrop selectors
    const overlaySelectors = [
      '.modal-backdrop',
      '.overlay',
      '.popup-overlay',
      '.modal-overlay',
      '.backdrop',
      '[data-testid="modal-backdrop"]',
      '[data-testid="overlay"]',
      '.ReactModal__Overlay',
      '.ant-modal-mask',
      '.el-overlay',
      '.v-overlay__scrim'
    ];
    
    for (const selector of overlaySelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          await element.click();
          logger.debug(`[Crawler][${jobId}] Clicked overlay: ${selector}`);
          await page.waitForTimeout(500);
          break;
        }
      } catch (error) {
        // Continue to next selector if this one fails
        continue;
      }
    }
    
    // If no overlay found, try clicking on document body coordinates that are likely empty
    await clickEmptyBodyArea(page, jobId);
    
  } catch (error) {
    logger.debug(`[Crawler][${jobId}] Click empty space failed: ${error}`);
  }
}

/**
 * Click on empty areas of the page body
 */
async function clickEmptyBodyArea(page: Page, jobId: string): Promise<void> {
  try {
    // Get viewport dimensions
    const viewport = page.viewport();
    if (!viewport) return;
    
    // Click on top-left corner (often empty)
    await page.mouse.click(10, 10);
    await page.waitForTimeout(300);
    
    // Click on top-right corner
    await page.mouse.click(viewport.width - 10, 10);
    await page.waitForTimeout(300);
    
    logger.debug(`[Crawler][${jobId}] Clicked empty body areas`);
  } catch (error) {
    logger.debug(`[Crawler][${jobId}] Click empty body area failed: ${error}`);
  }
}

/**
 * Handle specific types of popups commonly found on websites
 */
async function handleSpecificPopups(page: Page, jobId: string): Promise<void> {
  try {
    // LinkedIn login popup
    await handleLinkedInPopup(page, jobId);
    
    // Cookie consent banners
    await handleCookieBanners(page, jobId);
    
    // Newsletter signup popups
    await handleNewsletterPopups(page, jobId);
    
    // Age verification popups
    await handleAgeVerificationPopups(page, jobId);
    
  } catch (error) {
    logger.debug(`[Crawler][${jobId}] Specific popup handling failed: ${error}`);
  }
}

/**
 * Handle LinkedIn login popup specifically
 */
async function handleLinkedInPopup(page: Page, jobId: string): Promise<void> {
  try {
    // Check if we're on LinkedIn
    const url = page.url();
    const isLinkedIn = url.includes('linkedin.com');
    
    if (isLinkedIn) {
      logger.info(`[Crawler][${jobId}] Detected LinkedIn page, applying specific popup handling`);
      
      // Multiple attempts with ESC key first (most reliable for LinkedIn)
      await page.keyboard.press('Escape');
      await page.waitForTimeout(500);
      await page.keyboard.press('Escape');
      await page.waitForTimeout(500);
    }
    
    const linkedinSelectors = [
      '.modal__dismiss',
      '.modal__close-btn',
      '[data-test-modal-close-btn]',
      '.artdeco-modal__dismiss',
      '.artdeco-modal__dismiss-btn',
      '.msg-overlay-bubble-header__control--close-btn',
      '.authentication-outlet__dismiss-btn',
      '.modal__dismiss-btn',
      '.authwall-join-form__form-toggle--ignore',
      '.join-form__dismiss',
      '.guest-homepage-nav__join-cta',
      '[data-tracking-control-name="guest_homepage_basic_nav_header_signin"]',
      '.sign-in-modal__outlet-btn',
      '.modal-wormhole-content [aria-label="Dismiss"]',
      '.modal-wormhole-content .artdeco-button--tertiary',
      '.auth-wall__dismiss-btn'
    ];
    
    for (const selector of linkedinSelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          await element.click();
          logger.info(`[Crawler][${jobId}] Closed LinkedIn popup: ${selector}`);
          await page.waitForTimeout(1000); // Longer wait for LinkedIn
          break;
        }
      } catch (error) {
        continue;
      }
    }
    
    // If still LinkedIn, try clicking on the background overlay
    if (isLinkedIn) {
      try {
        const overlaySelectors = [
          '.modal-wormhole',
          '.modal-backdrop',
          '.artdeco-modal-overlay',
          '.modal__backdrop'
        ];
        
        for (const selector of overlaySelectors) {
          const element = await page.$(selector);
          if (element) {
            // Click on the top-left corner of the overlay (likely background)
            const box = await element.boundingBox();
            if (box) {
              await page.mouse.click(box.x + 10, box.y + 10);
              logger.info(`[Crawler][${jobId}] Clicked LinkedIn overlay background`);
              await page.waitForTimeout(1000);
              break;
            }
          }
        }
      } catch (error) {
        logger.debug(`[Crawler][${jobId}] LinkedIn overlay click failed: ${error}`);
      }
    }
  } catch (error) {
    logger.debug(`[Crawler][${jobId}] LinkedIn popup handling failed: ${error}`);
  }
}

/**
 * Handle cookie consent banners
 */
async function handleCookieBanners(page: Page, jobId: string): Promise<void> {
  try {
    const cookieSelectors = [
      '[data-testid="cookie-banner-close"]',
      '.cookie-banner-close',
      '.cookie-consent-close',
      '.gdpr-close',
      '.privacy-banner-close',
      '.cc-dismiss',
      '.cookielaw-close-button'
    ];
    
    for (const selector of cookieSelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          await element.click();
          logger.debug(`[Crawler][${jobId}] Closed cookie banner: ${selector}`);
          await page.waitForTimeout(500);
          break;
        }
      } catch (error) {
        continue;
      }
    }
  } catch (error) {
    logger.debug(`[Crawler][${jobId}] Cookie banner handling failed: ${error}`);
  }
}

/**
 * Handle newsletter signup popups
 */
async function handleNewsletterPopups(page: Page, jobId: string): Promise<void> {
  try {
    const newsletterSelectors = [
      '.newsletter-popup-close',
      '.email-signup-close',
      '.subscription-modal-close',
      '[data-testid="newsletter-close"]',
      '.popup-newsletter-close'
    ];
    
    for (const selector of newsletterSelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          await element.click();
          logger.debug(`[Crawler][${jobId}] Closed newsletter popup: ${selector}`);
          await page.waitForTimeout(500);
          break;
        }
      } catch (error) {
        continue;
      }
    }
  } catch (error) {
    logger.debug(`[Crawler][${jobId}] Newsletter popup handling failed: ${error}`);
  }
}

/**
 * Handle age verification popups
 */
async function handleAgeVerificationPopups(page: Page, jobId: string): Promise<void> {
  try {
    const ageVerificationSelectors = [
      '.age-verification-close',
      '.age-gate-close',
      '[data-testid="age-verification-close"]',
      '.adult-content-close'
    ];
    
    for (const selector of ageVerificationSelectors) {
      try {
        const element = await page.$(selector);
        if (element) {
          await element.click();
          logger.debug(`[Crawler][${jobId}] Closed age verification popup: ${selector}`);
          await page.waitForTimeout(500);
          break;
        }
      } catch (error) {
        continue;
      }
    }
  } catch (error) {
    logger.debug(`[Crawler][${jobId}] Age verification popup handling failed: ${error}`);
  }
}

/**
 * Wait for popups to appear and handle them
 */
export async function waitAndHandlePopups(page: Page, jobId: string, maxWaitTime: number = 3000): Promise<void> {
  try {
    // Wait for potential popups to appear
    await page.waitForTimeout(maxWaitTime);
    
    // Handle any popups that appeared
    await handlePopups(page, jobId);
    
  } catch (error) {
    logger.debug(`[Crawler][${jobId}] Wait and handle popups failed: ${error}`);
  }
}